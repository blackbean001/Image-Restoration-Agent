from langgraph.graph import StateGraph, END
import copy
from PIL import Image, ImageOps
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
from utils.util import *


tmp_input_dir = Path("tmp_img_input")
tmp_output_dir = Path("tmp_img_output")
final_output = Path("final_output")

os.makedirs(tmp_input_dir, exist_ok=True)
#os.makedirs(tmp_input_dir / "input", exist_ok=True)
os.makedirs(tmp_output_dir, exist_ok=True)
os.makedirs(final_output, exist_ok=True)


class ImageState(dict):
    input_img_path: Path
    image: Image.Image
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


def load_image(state: ImageState):
    img = Image.open(state["input_img_path"])
    state["image"] = img
    state["cur_path"] = state["input_img_path"]
    shutil.copy(state["input_img_path"], tmp_input_dir / "input.png")
    #shutil.copy(state["input_img_path"], tmp_input_dir / "input" / "input.png")
    print(f"Finished loading image from {state['input_img_path']}, and copy to {tmp_input_dir}")
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

    state["initial_plan"] = copy.deepcopy(plan)
    state["remaining_plan"] = plan
    print(f"Finished proposing plan with DepictQA: {state['initial_plan']}")
    return state


def propose_plan_retrieval(state: ImageState):
    # retrive
    # return [("super-resolution", "diffbir"), \
    #        ("haze", "dehaze-former"), ...]
    evaluation = state["res_seq_retrieved"]
    plan = [(item[0], item[1]) for item in evaluation]
    state["initial_plan"] = copy.deepcopy(plan)
    state["remaining_plan"] = plan
    print(f"Finished proposing plan with PostgreSQL retrieval: {state['initial_plan']}")
    return state


def execute_one_degradation(state:ImageState):
    remaining_plan = state["remaining_plan"]
    print(f"Remaining subtasks for execution: {state['remaining_plan']}")
    index = str(len(state["initial_plan"])-len(remaining_plan))

    subtask = remaining_plan.pop(0)

    # only one tool when using retrieval
    toolbox = get_toolbox(state, subtask)

    o_name = "_".join(str(state['input_img_path']).split("/")[-1:])
    
    processed_images = {item.split("-")[0]:None for item in os.listdir(tmp_output_dir)}
    if o_name in processed_images and index == 0:
        print(f"Image {o_name} has already been processed. SKIP...")
        skip = True
        exit()
    
    if index == "0":
        task_id = f"{o_name}-{strftime('%y%m%d_%H%M%S', localtime())}"
        state["task_id"] = task_id

        task_output_dir = os.path.join(os.path.abspath('.'), tmp_output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
    else:
        task_output_dir = os.path.join(os.path.abspath('.'), tmp_output_dir, state["task_id"])

    subtask_output_dir = Path(task_output_dir) / (subtask + "-" + index)
    os.makedirs(subtask_output_dir, exist_ok=True)
    print(f"Generate output to {subtask_output_dir}")

    res_degra_level_dict: dict[str, list[Path]] = {}
    success = True
    best_img_path = state["cur_path"]

    for tool in toolbox:
        tool_output_dir = subtask_output_dir / tool.tool_name
        os.makedirs(tool_output_dir, exist_ok=True)
        tool(
            input_dir=os.path.abspath('.') / tmp_input_dir,
            output_dir=tool_output_dir,
            silent=True,
        )
        state["cur_path"] = tool_output_dir / "output.png"
        degra_level = evaluate_tool_result(state)
        print(f"Using tool {tool.tool_name}, output to {tool_output_dir}, degra_level {degra_level}")
        res_degra_level_dict.setdefault(degra_level, []).append(state["cur_path"])
        if degra_level == "very low":
            res_degra_level = "very low"
            best_tool_name = tool.tool_name
            state["best_img_path"] = tool_output_dir / "output.png"
            shutil.copy(tool_output_dir / "output.png", tmp_input_dir / "input.png")
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
                shutil.copy(best_img_path, tmp_input_dir / "input.png")
                state["best_img_path"] = best_img_path
                break

    state["remaining_plan"] = remaining_plan
    state["subtask_success"][subtask + "-" + index] = success
    return state


def get_output(state:ImageState):
    shutil.copy(state["best_img_path"], "final_output")    


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


# input args
invoke_dict = {}

AgenticIR_dir = Path("/home/jason/Auto-Image-Restoration-Service/Auto-Image-Restoration/AgenticIR")
CLIP4CIR_model_dir = Path("/home/jason/CLIP4Cir/models")

invoke_dict["input_img_path"] = "/home/jason/Auto-Image-Restoration-Service/Auto-Image-Restoration/AgentApp/100.png"

invoke_dict["depictqa"] = get_depictqa()
invoke_dict["gpt4"] = get_GPT4(AgenticIR_dir / "config.yml")

invoke_dict["levels"] = ["very low", "low", "medium", "high", "very high"]
invoke_dict["schedule_experience_path"] = AgenticIR_dir / "memory/schedule_experience.json"

invoke_dict["retrieval_args"] = {}
invoke_dict["retrieval_args"]["combining_function"] = "combiner"
invoke_dict["retrieval_args"]["combiner_path"] = CLIP4CIR_model_dir / "combiner_trained_on_imgres_RN50x4_2025-09-05_12:30:03/saved_models/combiner_arithmetic.pt"
invoke_dict["retrieval_args"]["clip_model_name"] = "RN50x4"
invoke_dict["retrieval_args"]["clip_model_path"] = CLIP4CIR_model_dir / "clip_finetuned_on_imgres_RN50x4_2025-09-05_10:48:31/saved_models/tuned_clip_arithmetic.pt"
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
invoke_dict['with_experience'] = True
invoke_dict['subtask_success'] = {}
invoke_dict['task_id'] = ""

# compile and run
app = workflow.compile()
result = app.invoke(invoke_dict)



