from langgraph.graph import StateGraph, END
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
os.makedirs(tmp_output_dir, exist_ok=True)
os.makedirs(final_output, exist_ok=True)


class ImageState(dict):
    input_img_path: PATH,
    image: Image.Image,
    retrieved_img_name: str,
    res_seq_retrieved: str,
    res_seq_depictqa: str,
    sim: float,
    desc: str,
    remaining_task: list,
    initial_plan: list,
    cur_plan: list,
    cur_path: str,
    depictqa: DepictQA,
    best_img_path: str,
    retrieval_args: dict,
    qa_logger: logging.Logger


def init_agent(state: ImageState):
    state["depictqa"] = get_depictqa()
    state["levels"] = ["very low", "low", "medium", "high", "very high"]      
    state["schedule_experience_path"] = "memory/schedule_experience.json"

    state["retrieval_args"] = {}
    state["retrieval_args"]["combining_function"] = "combiner"
    state["retrieval_args"]["combiner_path"] = "/home/jason/CLIP4Cir/models/combiner_trained_on_imgres_RN50x4_2025-09-05_12:30:03/saved_models/combiner_arithmetic.pt"
    state["retrieval_args"]["clip_model_name"] = "RN50x4"
    state["retrieval_args"]["clip_model_path"] = "/home/jason/CLIP4Cir/models/clip_finetuned_on_imgres_RN50x4_2025-09-05_10:48:31/saved_models/tuned_clip_arithmetic.pt"
    state["retrieval_args"]["projection_dim"] = 2560
    state["retrieval_args"]["hidden_dim"] = 5120
    state["retrieval_args"]["transform"] = "targetpad"
    state["retrieval_args"]["target_ratio"] = 1.25
    
    state["degra_subtask_dict"] = {
                    "low resolution": "super-resolution",
                    "noise": "denoising",
                    "motion blur": "motion deblurring",
                    "defocus blur": "defocus deblurring",
                    "haze": "dehazing",
                    "rain": "deraining",
                    "dark": "brightening",
                    "jpeg compression artifact": "jpeg compression artifact removal"}
    state["subtask_degra_dict"] = {
            v: k for k, v in degra_subtask_dict.items()}
    state["degradations"] = set(degra_subtask_dict.keys())
    state["subtasks"] = set(degra_subtask_dict.values())


def load_image(state: ImageState):
    img = Image.open(state["input_img_path"])
    state["image"] = img
    state["cur_path"] = state["input_img_path"]
    shutil.copy(state["input_img_path"], tmp_input_dir / "input.png")
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

    return state


# [('motion blur', 'very high'), ...]
def first_evaluate_by_depictqa(state: ImageState):
    evaluation = eval(
        depictqa(Path(state["input_img_path"]), task="eval_degradation"))
    
    state["res_seq_depictqa"] = evaluation
    
    return state


def evaluate_tool_result(state: ImageState):
    level = eval(
        depictqa(Path(state["cur_path"]), task="eval_degradation")
        )[0][1]
    return level


def propose_plan_depictqa(state: ImageState):
    # retrieval threshold not reached, use depictqa
    # return ["super-resolution", "haze", ...]
    img_shape = state["image"].size
    if max(img_shape) < 300:
        agenda.append("super-resolution")
    for degradation, severity in state["res_seq_depictqa"]:
        if state["levels"].index(severity) >= 2:
            agenda.append(state["degra_subtask_dict"]["degradation"])
    random.shuffle(agenda)
    if len(agenda) <= 1:
        state['initial_plan'] = agenda
        state['cur_plan'] = agenda
        return state

    degradations = [subtask_degra_dict[subtask] for subtask in agenda]
    if self.with_retrieval:
        plan = schedule_w_retrieval(state, gpt4, degradations, agenda, ps)
    else:
        plan = schedule_wo_retrieval(state, gpt4, degradations, agenda, ps)

    state["initial_plan"] = plan
    state["cur_plan"] = plan
    return state


def propose_plan_retrieval(state: ImageState):
    # retrive
    # return [("super-resolution", "diffbir"), \
    #        ("haze", "dehaze-former"), ...]
    evaluation = state["res_seq_retrieved"]
    plan = [(item[0], item[1]) for item in evaluation]
    state["initial_plan"] = plan
    state["cur_plan"] = plan
    return state


def execute_one_degradation(state:ImageState):
    cur_plan = state["cur_plan"]
    subtask = cur_plan.pop(0)
    # only one tool when using retrieval
    toolbox = get_toolbox(subtask)
    
    o_name = "_".join(str(state['path']).split("/")[-2:])
    
    processed_images = {item.split("-")[0]:None for item in os.listdir(tmp_output_dir)}
    if o_name in processed_images:
        print(f"Image {o_name} has already been processed. SKIP...")
        skip = True
        exit()

    task_id = f"{o_name}-{strftime('%y%m%d_%H%M%S', localtime())}"

    output_dir = tmp_output_dir / task_id
    os.makedirs(output_dir, exist_ok=True)

    success = True
    for tool in toolbox:
        output_dir = output_dir / tool.name
        os.makedirs(output_dir, exist_ok=True)
        tool(
            input_dir=tmp_input_dir,
            output_dir=output_dir,
            silent=True,
        )
        state["cur_path"] = output_dir / "output.png"
        degra_level = evaluate_tool_result(state)
        res_degra_level_dict.setdefault(degra_level, []).append(state["cur_path"])
        if degra_level == "very low":
            res_degra_level = "very low"
            best_tool_name = tool.tool_name
            state["best_img_pth"] = output_dir / "output.png"
            shutil.copy(output_dir / "output.png", tmp_input_dir / "input.png")
            break
    # no result with "very low" degradation level
    else:
        for res_level in state["levels"][1:]:
            if res_level in res_degra_level_dict:
                candidates = res_degra_level_dict[res_level]
                best_img_path = search_best_by_comp(candidates, state)
                best_tool_name = best_img_path.parents[0].name
                success = False if res_level != "low"
                res_degra_level = res_level
                break

    if success:
        state["best_img_path"] = best_img_path
        shutil.copy(best_img_path, tmp_input_dir / "input.png")
    
    state["cur_plan"] = cur_plan
    return success


def get_output(state:ImageState):
    shutil.copy(state["best_img_path"], "final_output")    


# define workflow
workflow = StateGraph(ImageState)

workflow.add_node("init_agent", init_agent)
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
    if len(state["cur_plan"]) == 0:
        return "finish"
    else:
        return "continue"


workflow.set_entry_point("init_agent")
workflow.add_edge("init_agent", "load_image")
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


# compile and run
app = workflow.compile()
result = app.invoke({"path": "/home/jason/Auto-Image-Restoration-Service/Auto-Image-Restoration/AgentApp/001.png"})









