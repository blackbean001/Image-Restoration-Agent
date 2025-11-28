from langgraph.graph import StateGraph, END
from copy import deepcopy as cpy
from PIL import Image, ImageOps
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
from utils.util import *


ROOT = "/home/jason/Auto-Image-Restoration"


# define LangGraph state
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
    #desc: str
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
    assert state["input_img_path"] != "", "Please input image_path or image"
    if state["image"] == None:
        img = Image.open(state["input_img_path"])
        state["image"] = img
    state["cur_path"] = cpy(state["input_img_path"])

    shutil.copy(state["input_img_path"], state["tmp_input_dir"] / "input.png")
    print(f"Finished loading image from {state['input_img_path']}, and copy to {state['tmp_input_dir']}")
    return state


# [('motion blur', 'very high', 'hdrnet'), ...]
def evaluate_by_retrieval(state: ImageState):
    # generate combined embedding
    embedding = generate_retrieval_embedding(state)
    # retrieve result from database
    results = retrieve_from_database(embedding, 1)

    _id, name, res_seq, sim = results[0]
    
    evaluation = [(item.split("_")[0], 'very high', item.split("_")[1]) \
            for item in res_seq.split("/")]
    
    state["retrieved_img_name"] = name
    state["res_seq_retrieved"] = evaluation
    state["sim"] = float(sim)
    print(f"Finished image retrieval from PostgreSQL, max similarity {state['sim']}")

    return state


# [('motion blur', 'very high'), ...]
def first_evaluate_by_depictqa(state: ImageState):
    evaluation = eval(
        state["depictqa"](Path(state["input_img_path"]), task="eval_degradation"))
    
    state["res_seq_depictqa"] = evaluation
    print(f'Finished evaluate by DepictQA, result: {state["res_seq_depictqa"]}')

    return state


def evaluate_tool_result(state: ImageState):
    print("Evaluate img: ", state["cur_path"])
    level = eval(
        state["depictqa"](Path(state["cur_path"]), task="eval_degradation")
        )[0][1]
    return level


def propose_plan_depictqa(state: ImageState):
    # retrieval threshold not reached, use depictqa
    # return ["super-resolution", "haze", ...]
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

    degradations = [subtask_degra_dict[subtask] for subtask in agenda]
    if state["with_experience"]:
        plan = schedule_w_experience(state, state["gpt4"], degradations, agenda, "")
    else:
        plan = schedule_wo_experience(state, state["gpt4"], degradations, agenda, "")

    state["initial_plan"] = cpy(plan)
    state["remaining_plan"] = cpy(plan)
    print(f"Finished proposing plan with DepictQA: {state['initial_plan']}")
    return state


def propose_plan_retrieval(state: ImageState):
    # retrive
    # return [("super-resolution", "diffbir"), \
    #        ("haze", "dehaze-former"), ...]
    evaluation = state["res_seq_retrieved"]
    plan = [(item[0], item[1]) for item in evaluation]
    state["initial_plan"] = cpy(plan)
    state["remaining_plan"] = cpy(plan)
    print(f"Finished proposing plan with PostgreSQL retrieval: {state['initial_plan']}")
    return state


def execute_one_degradation(state:ImageState):
    remaining_plan = state["remaining_plan"]
    state["executed_plans"].append(cpy(remaining_plan))
    print(f"Remaining subtasks for execution: {remaining_plan}")
    print(f"Executed_plans: ", state['executed_plans'])

    index = str(state["tool_execution_count"])

    subtask = remaining_plan.pop(0)

    if subtask != "brightening":
        toolbox = list(model_service_yaml[subtask].keys())
    else:
        toolbox = ["constant_shift", "gamma_correction", "histogram_equalization"]
    # toolbox = get_toolbox(state, subtask)

    o_name = "_".join(str(state['input_img_path']).split("/")[-1:])

    processed_images = {item.split("-")[0]:None for item in os.listdir(state['tmp_output_dir'])}
    if o_name in processed_images and index == 0:
        print(f"Image {o_name} has already been processed. SKIP...")
        skip = True
        exit()

    if index == "0":
        task_id = f"{o_name}-{strftime('%y%m%d_%H%M%S', localtime())}"
        state["task_id"] = task_id

        task_output_dir = os.path.join(os.path.abspath('.'), state['tmp_output_dir'], task_id)
        os.makedirs(task_output_dir, exist_ok=True)
    else:
        task_output_dir = os.path.join(os.path.abspath('.'), state['tmp_output_dir'], state["task_id"])

    subtask_output_dir = Path(task_output_dir) / (subtask + "-" + index)
    os.makedirs(subtask_output_dir, exist_ok=True)
    print(f"Generate output to {subtask_output_dir}")

    res_degra_level_dict: dict[str, list[Path]] = {}
    success = True
    best_img_path = cpy(state["cur_path"])

    for tool in toolbox:
        tool_yml = model_service_yaml[subtask][tool]
        
        tool_name = tool
        DIR = tool_yml["dir"]
        host = tool_yml["host"]
        port = tool_yml["port"]
        request_cmd = tool_yml["curl_cmd"]
        server_launch_cmd = f"nohup conda run {tool_name.lower()} python {os.path.join(ROOT, DIR, 'model_serving.py)'} >> logs/logs_{tool_name}.log"

        in_use = is_port_in_use(port, host)
        
        # launch server
        if not in_use:
            os.system(server_launch_cmd)

        tool_output_dir = subtask_output_dir / tool_name
        os.makedirs(tool_output_dir, exist_ok=True)

        # request
        if subtask != "brightening":
            request_cmd.replace("input_path", state["input_img_path"]).replace("output_path", tool_output_dir / "output.png")
        else:
            request_cmd.replace("brightening_method", "constant_shift")

        os.system(request_cmd)

        state["cur_path"] = tool_output_dir / "output.png"
        
        # evaluate result
        degra_level = evaluate_tool_result(state)
        print(f"Using tool {tool.tool_name}, output to {tool_output_dir}, degra_level {degra_level}")
        
        res_degra_level_dict.setdefault(degra_level, []).append(state["cur_path"])
        if degra_level == "very low":
            res_degra_level = "very low"
            best_tool_name = tool.tool_name
            state["best_img_path"] = tool_output_dir / "output.png"
            shutil.copy(tool_output_dir / "output.png", state['tmp_input_dir'] / "input.png")
            #shutil.copy(output_dir / "output.png", tmp_input_dir / "input" / "input.png")
            print(f"Finished subtask {subtask} by execution best tool: {best_tool_name}")
            break

    # no result with "very low" degradation level
    else:
        for res_level in state["levels"][1:]:
            if res_level in res_degra_level_dict:
                candidates = res_degra_level_dict[res_level]
                best_img_path = search_best_by_comp(candidates, state)
                best_tool_name = best_img_path.parents[0].name
                if res_level == "low":
                    success = True
                else:
                    success = False
                print(f"Finished subtask by comparing, success: {success},  \
                        best_img_path: {best_img_path}, best_tool_name: {best_tool_name}")
                res_degra_level = res_level
                shutil.copy(best_img_path, state['tmp_input_dir'] / "input.png")
                state["best_img_path"] = best_img_path
                break
    
    state["remaining_plan"] = remaining_plan
    state["subtask_success"][subtask + "-" + index] = success
    state["tool_execution_count"] += 1

    # rollback if not success
    if not success and state["with_rollback"]:
        roll_back_plans = state["remaining_plan"] + [subtask]
        print(f"Plan {subtask} is not successful, thus need to rollback...")
        if roll_back_plans not in state["executed_plans"]:
            state["remaining_plan"] = roll_back_plans
            print(f"Subtask ({subtask}) restoration not success, insert baack to the plans, current remaining plans: {remaining_plan}")
    else:
        print(f"Plan {subtask} execute successfully.")
    return state


def get_output(state:ImageState):
    shutil.copy(state["best_img_path"], "final_output")
    print("Finished image restoration, output to ./final_output")
    return state


    return state


def get_output(state:ImageState):
    shutil.copy(state["best_img_path"], "final_output")    
    print("Finished image restoration, output to ./final_output")
    return state


# define workflow
workflow = StateGraph(ImageState)

#workflow.add_node("init_agent", init_agent)
workflow.add_node("load_image", load_image)
workflow.add_node("evaluate_by_retrieval", evaluate_by_retrieval)
workflow.add_node("first_evaluate_by_depictqa", first_evaluate_by_depictqa)
workflow.add_node("evaluate_tool_result", evaluate_tool_result)
workflow.add_node("propose_plan_retrieval", propose_plan_retrieval)
workflow.add_node("propose_plan_depictqa", propose_plan_depictqa)
workflow.add_node("execute_one_degradation", execute_one_degradation)
workflow.add_node("get_output", get_output)


# add edges
# function to decide whether to use retrieval
def use_retrieval(state:ImageState):
    if state["sim"] >= 0.9:
        return "use_retrieval"
    else:
        return "use_depictqa"


def plan_state(state:ImageState):
    if len(state["remaining_plan"]) == 0:
        return "finish"
    else:
        return "continue"

def create_image_analysis_graph():
    #workflow.set_entry_point("init_agent")
    #workflow.add_edge("init_agent", "load_image")
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


def run_agent():
    # input args
    invoke_dict = {}

    AgenticIR_dir = Path("../AgenticIR")
    CLIP4CIR_model_dir = Path("../AgenticIR/retrival_database/CLIP4CIR/models")

    # set input_img_path
    invoke_dict["input_img_path"] = "./demo_input/input.png"
    invoke_dict["model_service_yaml"] = "./model_service/model_services.yaml"

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

    invoke_dict["degra_subtask_dict"] = {
                    "low resolution": "super-resolution",
                    "noise": "denoising",
                    "motion blur": "motion deblurring",
                    "defocus blur": "defocus deblurring",
                    "haze": "dehazing",
                    "rain": "deraining",
                    "dark": "brightening",
                    "jpeg compression artifact": "jpeg compression artifact removal"}
    invoke_dict["subtask_degra_dict"] = {
                    v: k for k, v in degra_subtask_dict.items()}
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
    
    os.makedirs(invoke_dict["tmp_input_dir"], exist_ok=True)
    os.makedirs(invoke_dict["tmp_output_dir"], exist_ok=True)

    shutil.copy(invoke_dict["input_img_path"], invoke_dict["tmp_input_dir"])

    # compile and run
    app = create_image_analysis_graph()
    result = app.invoke(invoke_dict)

if __name__ == "__main__":
    run_agent()




