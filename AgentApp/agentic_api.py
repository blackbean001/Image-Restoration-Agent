import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from flask import Flask, request, jsonify, send_file
from langgraph.graph import StateGraph, END
from copy import deepcopy as cpy
from PIL import Image, ImageOps
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
from utils.util import *
import traceback

app = Flask(__name__)


# Define LangGraph state
class ImageState(dict):
    input_img_path: Path
    image: Image.Image
    tmp_input_dir: Path
    tmp_output_dir: Path
    final_output: Path
    depictqa: DepictQA
    gpt4: GPT4
    levels: List
    schedule_experience_path: str
    retrieval_args: dict
    retrieved_img_name: str
    res_seq_retrieved: str
    res_seq_depictqa: str
    sim: float
    degra_subtask_dict: dict
    subtask_degra_dict: dict
    degradations: set
    subtasks: set
    with_experience: bool
    initial_plan: list
    remaining_plan: list
    cur_path: str
    best_img_path: str
    subtask_success: dict
    task_id: str
    qa_logger: logging.Logger
    with_rollback: bool
    tool_execution_count: int
    executed_plans: list


def load_image(state: ImageState):
    """Load input image and initialize state"""
    assert state["input_img_path"] != "", "Please input image_path or image"
    if state["image"] == None:
        img = Image.open(state["input_img_path"])
        state["image"] = img
    state["cur_path"] = cpy(state["input_img_path"])

    shutil.copy(state["input_img_path"], state["tmp_input_dir"] / "input.png")
    print(f"Finished loading image from {state['input_img_path']}, and copy to {state['tmp_input_dir']}")
    return state


def evaluate_by_retrieval(state: ImageState):
    """Evaluate image degradation using retrieval from database"""
    embedding = generate_retrieval_embedding(state)
    results = retrieve_from_database(embedding, 1)

    _id, name, res_seq, sim = results[0]

    evaluation = [(item.split("_")[0], 'very high', item.split("_")[1]) \
            for item in res_seq.split("/")]

    state["retrieved_img_name"] = name
    state["res_seq_retrieved"] = evaluation
    state["sim"] = float(sim)
    print(f"Finished image retrieval from PostgreSQL, max similarity {state['sim']}")

    return state


def first_evaluate_by_depictqa(state: ImageState):
    """First evaluation using DepictQA model"""
    evaluation = eval(
        state["depictqa"](Path(state["input_img_path"]), task="eval_degradation"))

    state["res_seq_depictqa"] = evaluation
    print(f'Finished evaluate by DepictQA, result: {state["res_seq_depictqa"]}')

    return state


def evaluate_tool_result(state: ImageState):
    """Evaluate the result of a tool execution"""
    print("Evaluate img: ", state["cur_path"])
    level = eval(
        state["depictqa"](Path(state["cur_path"]), task="eval_degradation")
        )[0][1]
    return level


def propose_plan_depictqa(state: ImageState):
    """
    Propose restoration plan based on DepictQA evaluation
    Returns a list of subtasks like ["super-resolution", "dehazing", ...]
    """
    agenda = []
    img_shape = state["image"].size
    if max(img_shape) < 300:
        agenda.append("super-resolution")
    for degradation, severity in state["res_seq_depictqa"]:
        if state["levels"].index(severity) >= 2:
            agenda.append(state["degra_subtask_dict"][degradation])
    random.shuffle(agenda)
    if len(agenda) <= 1:
        state['initial_plan'] = agenda
        state['remaining_plan'] = agenda
        return state

    degradations = [state["subtask_degra_dict"][subtask] for subtask in agenda]
    if state["with_experience"]:
        plan = schedule_w_experience(state, state["gpt4"], degradations, agenda, "")
    else:
        plan = schedule_wo_experience(state, state["gpt4"], degradations, agenda, "")

    state["initial_plan"] = cpy(plan)
    state["remaining_plan"] = cpy(plan)
    print(f"Finished proposing plan with DepictQA: {state['initial_plan']}")
    return state


def propose_plan_retrieval(state: ImageState):
    """
    Propose restoration plan based on retrieval results
    Returns a list of tuples like [("motion blur", "hdrnet"), ...]
    """
    evaluation = state["res_seq_retrieved"]
    plan = [(item[0], item[1]) for item in evaluation]
    state["initial_plan"] = cpy(plan)
    state["remaining_plan"] = cpy(plan)
    print(f"Finished proposing plan with PostgreSQL retrieval: {state['initial_plan']}")
    return state


def execute_one_degradation(state:ImageState):
    """Execute one degradation restoration subtask"""
    remaining_plan = state["remaining_plan"]
    state["executed_plans"].append(cpy(remaining_plan))
    print(f"Remaining subtasks for execution: {remaining_plan}")
    print(f"Executed_plans: ", state['executed_plans'])

    index = str(state["tool_execution_count"])

    subtask = remaining_plan.pop(0)

    # Get toolbox for current subtask (only one tool when using retrieval)
    toolbox = get_toolbox(state, subtask)

    o_name = "_".join(str(state['input_img_path']).split("/")[-1:])

    # Check if image has already been processed
    processed_images = {item.split("-")[0]:None for item in os.listdir(state['tmp_output_dir'])}
    if o_name in processed_images and index == "0":
        print(f"Image {o_name} has already been processed. SKIP...")
        return state

    # Create task directory on first execution
    if index == "0":
        task_id = f"{o_name}-{strftime('%y%m%d_%H%M%S', localtime())}"
        state["task_id"] = task_id

        task_output_dir = os.path.join(os.path.abspath('.'), state['tmp_output_dir'], task_id)
        os.makedirs(task_output_dir, exist_ok=True)
    else:
        task_output_dir = os.path.join(os.path.abspath('.'), state['tmp_output_dir'], state["task_id"])

    # Create subtask output directory
    subtask_output_dir = Path(task_output_dir) / (subtask + "-" + index)
    os.makedirs(subtask_output_dir, exist_ok=True)
    print(f"Generate output to {subtask_output_dir}")

    res_degra_level_dict: dict[str, list[Path]] = {}
    success = True
    best_img_path = cpy(state["cur_path"])
    state["best_img_path"] = best_img_path

    # Execute all tools in the toolbox
    for tool in toolbox:
        tool_output_dir = subtask_output_dir / tool.tool_name
        os.makedirs(tool_output_dir, exist_ok=True)
        try:
            tool(
                input_dir=os.path.abspath('.') / state['tmp_input_dir'],
                output_dir=tool_output_dir,
                silent=True,
            )
        except Exception as e:
            print(f"Error: {e}")

        state["cur_path"] = tool_output_dir / "output.png"
        degra_level = evaluate_tool_result(state)
        print(f"Using tool {tool.tool_name}, output to {tool_output_dir}, degra_level {degra_level}")
        res_degra_level_dict.setdefault(degra_level, []).append(state["cur_path"])
        
        # If degradation level is very low, task succeeds
        if degra_level == "very low":
            res_degra_level = "very low"
            best_tool_name = tool.tool_name
            state["best_img_path"] = tool_output_dir / "output.png"
            shutil.copy(tool_output_dir / "output.png", state['tmp_input_dir'] / "input.png")
            print(f"Finished subtask {subtask} by execution best tool: {best_tool_name}")
            break
    # No result with "very low" degradation level found
    else:
        # Find best result from other degradation levels
        for res_level in state["levels"][1:]:
            if res_level in res_degra_level_dict:
                candidates = res_degra_level_dict[res_level]
                best_img_path = search_best_by_comp(candidates, state)
                best_tool_name = best_img_path.parents[0].name
                if res_level == "low":
                    success = True
                else:
                    success = False
                print(f"Finished subtask by comparing, success: {success}, best_img_path: {best_img_path}, best_tool_name: {best_tool_name}")
                res_degra_level = res_level
                shutil.copy(best_img_path, state['tmp_input_dir'] / "input.png")
                state["best_img_path"] = best_img_path
                break

    state["remaining_plan"] = remaining_plan
    state["subtask_success"][subtask + "-" + index] = success
    state["tool_execution_count"] += 1

    # Rollback if not successful
    if not success and state["with_rollback"]:
        roll_back_plans = state["remaining_plan"] + [subtask]
        print(f"Plan {subtask} is not successful, thus need to rollback...")
        if roll_back_plans not in state["executed_plans"]:
            state["remaining_plan"] = roll_back_plans
            print(f"Subtask ({subtask}) restoration not success, insert back to the plans, current remaining plans: {remaining_plan}")
    else:
        print(f"Plan {subtask} execute successfully.")
    
    return state


def get_output(state:ImageState):
    """Save final output image"""
    output_path = state["final_output"] / f"{state['task_id']}.png"
    try:
        shutil.copy(state["best_img_path"], output_path)
        print(f"Finished image restoration, output to {output_path}")
        state["final_output_path"] = str(output_path)
        return state
    except Exception as e:
        print(f"Failed get_output, Error: {e}")
        return


def use_retrieval(state:ImageState):
    """Decide whether to use retrieval or DepictQA based on similarity score"""
    if state["sim"] >= 0.9:
        return "use_retrieval"
    else:
        return "use_depictqa"


def plan_state(state:ImageState):
    """Check if there are remaining tasks in the plan"""
    if len(state["remaining_plan"]) == 0:
        return "finish"
    else:
        return "continue"


def create_image_analysis_graph():
    """Create and compile the LangGraph workflow"""
    workflow = StateGraph(ImageState)
    
    workflow.add_node("load_image", load_image)
    workflow.add_node("evaluate_by_retrieval", evaluate_by_retrieval)
    workflow.add_node("first_evaluate_by_depictqa", first_evaluate_by_depictqa)
    workflow.add_node("evaluate_tool_result", evaluate_tool_result)
    workflow.add_node("propose_plan_retrieval", propose_plan_retrieval)
    workflow.add_node("propose_plan_depictqa", propose_plan_depictqa)
    workflow.add_node("execute_one_degradation", execute_one_degradation)
    workflow.add_node("get_output", get_output)

    workflow.set_entry_point("load_image")
    workflow.add_edge("load_image", "evaluate_by_retrieval")
    workflow.add_conditional_edges(
        "evaluate_by_retrieval",
        use_retrieval,
        {
            "use_retrieval": "propose_plan_retrieval",
            "use_depictqa": "first_evaluate_by_depictqa",
        }
    )
    workflow.add_edge("first_evaluate_by_depictqa", "propose_plan_depictqa")
    workflow.add_edge("propose_plan_depictqa", "execute_one_degradation")
    workflow.add_edge("propose_plan_retrieval", "execute_one_degradation")
    workflow.add_conditional_edges(
        "execute_one_degradation",
        plan_state,
        {
            "finish": "get_output",
            "continue": "execute_one_degradation"
        }
    )
    workflow.add_edge("get_output", END)
    
    return workflow.compile()


def init_invoke_dict(input_img_path):
    invoke_dict = {}

    AgenticIR_dir = Path("../AgenticIR")
    CLIP4CIR_model_dir = Path("../AgenticIR/retrival_database/CLIP4CIR/models")

    invoke_dict["input_img_path"] = input_img_path
    invoke_dict["image"] = None

    invoke_dict["depictqa"] = get_depictqa()
    invoke_dict["gpt4"] = get_GPT4(AgenticIR_dir / "config.yml")

    invoke_dict["levels"] = ["very low", "low", "medium", "high", "very high"]
    invoke_dict["schedule_experience_path"] = AgenticIR_dir / "memory/schedule_experience.json"

    invoke_dict["retrieval_args"] = {}
    invoke_dict["retrieval_args"]["combining_function"] = "combiner"
    invoke_dict["retrieval_args"]["combiner_path"] = CLIP4CIR_model_dir / "combiner_trained_on_imgres_RN50x4/saved_models/combiner_arithmetic.pt"
    invoke_dict["retrieval_args"]["clip_model_name"] = "RN50x4"
    invoke_dict["retrieval_args"]["clip_model_path"] = CLIP4CIR_model_dir / "clip_finetuned_on_imgres_RN50x4/saved_models/tuned_clip_arithmetic.pt"
    invoke_dict["retrieval_args"]["projection_dim"] = 2560
    invoke_dict["retrieval_args"]["hidden_dim"] = 5120
    invoke_dict["retrieval_args"]["transform"] = "targetpad"
    invoke_dict["retrieval_args"]["target_ratio"] = 1.25

    degra_subtask_dict = {
        "low resolution": "super-resolution",
        "noise": "denoising",
        "motion blur": "motion deblurring",
        "defocus blur": "defocus deblurring",
        "haze": "dehazing",
        "rain": "deraining",
        "dark": "brightening",
        "jpeg compression artifact": "jpeg compression artifact removal"
    }
    
    invoke_dict["degra_subtask_dict"] = degra_subtask_dict
    invoke_dict["subtask_degra_dict"] = {v: k for k, v in degra_subtask_dict.items()}
    invoke_dict["all_degradations"] = set(degra_subtask_dict.keys())
    invoke_dict["all_subtasks"] = set(degra_subtask_dict.values())
    invoke_dict["with_experience"] = True
    invoke_dict["with_rollback"] = True
    invoke_dict["tmp_input_dir"] = Path("tmp_img_input")
    invoke_dict["tmp_output_dir"] = Path("tmp_img_output")
    invoke_dict["final_output"] = Path("final_output")
    invoke_dict["subtask_success"] = {}
    invoke_dict["task_id"] = ""
    invoke_dict["tool_execution_count"] = 0
    invoke_dict["executed_plans"] = []
    
    try:
        shutil.rmtree(invoke_dict["tmp_input_dir"])
        shutil.rmtree(invoke_dict["tmp_output_dir"])
    except:
        1
    os.makedirs(invoke_dict["tmp_input_dir"], exist_ok=True)
    os.makedirs(invoke_dict["tmp_output_dir"], exist_ok=True)
    os.makedirs(invoke_dict["final_output"], exist_ok=True)

    return invoke_dict


_compiled_graph = None

def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_image_analysis_graph()
    return _compiled_graph


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/process', methods=['POST'])
def process_image():
    try:
        image_path = None
        
        if request.is_json:
            data = request.get_json()
            image_path = data.get('image_path')
        
        elif 'image_path' in request.form:
            image_path = request.form['image_path']
        
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename:
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                image_path = upload_dir / file.filename
                file.save(image_path)
                image_path = str(image_path)
        
        if not image_path:
            return jsonify({
                "error": "No image_path provided or file uploaded"
            }), 400
        
        if not os.path.exists(image_path):
            return jsonify({
                "error": f"Image file not found: {image_path}"
            }), 404
        
        invoke_dict = init_invoke_dict(image_path)
        
        app_graph = get_compiled_graph()
        result = app_graph.invoke(invoke_dict)
        
        return jsonify({
            "status": "success",
            "task_id": result.get("task_id", ""),
            "output_path": result.get("final_output_path", ""),
            "initial_plan": result.get("initial_plan", []),
            "subtask_success": result.get("subtask_success", {})
        }), 200
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/download/<task_id>', methods=['GET'])
def download_result(task_id):
    try:
        output_path = Path("final_output") / f"{task_id}.png"
        if output_path.exists():
            return send_file(output_path, mimetype='image/png')
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1146, debug=True)
