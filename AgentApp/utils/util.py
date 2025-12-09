import sys
sys.path.append("../../AgenticIR")

import os
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
import cv2
import json
import random
import yaml
from typing import Optional

import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector

import clip
from clip.model import CLIP

from pipeline.insert_emb_to_postgresql import *
import pipeline.prompts as prompts
from llm import GPT4, DepictQA
from executor import executor, Tool
import socket


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        result = s.connect_ex((host, port))
        return result == 0


def get_logger(logger_name: str,
               log_file: Optional[Path | str] = None,
               console_log_level: int = logging.INFO,
               file_log_level: int = logging.INFO,
               console_format_str: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
               file_format_str: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
               silent: bool = False
               ) -> logging.Logger:
    """Gets a logger with the specified setting.

    Args:
        logger_name (str): Name of the logger.
        log_file (Path, optional): If not None, logs to this file. Defaults to None.
        console_log_level/file_log_level (int, optional): Logging level for console/file. One of logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL. Defaults to logging.INFO.
        console_format_str/file_format_str (str, optional): Format of the log message for console/file. Defaults to '%(asctime)s - %(levelname)s - %(name)s - %(message)s'.
        silent (bool, optional): If True, does not log to console. Defaults to False.

    Returns:
        logging.Logger: Logger object.
    """
    logger_id = f"{logger_name}@{time.time()}"
    logger = logging.getLogger(logger_id)
    logger.setLevel(min(console_log_level, file_log_level))
    if not silent:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = ColoredFormatter(console_format_str)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(file_format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


qa_logger = get_logger(
        logger_name="IRAgent QA",
        log_file= "logs/llm_qa.md",
        console_log_level=logging.WARNING,
        file_format_str="%(message)s",
        silent=True)


workflow_format_str = "%(asctime)s - %(levelname)s\n%(message)s\n"
workflow_logger: logging.Logger = get_logger(
    logger_name="IRAgent Workflow",
    log_file="logs/workflow.md",
    console_format_str=workflow_format_str,
    file_format_str=workflow_format_str,
    silent=True)

degra_subtask_dict = {
    "low resolution": "super-resolution",
    "noise": "denoising",
    "motion blur": "motion deblurring",
    "defocus blur": "defocus deblurring",
    "haze": "dehazing",
    "rain": "deraining",
    "dark": "brightening",
    "jpeg compression artifact": "jpeg compression artifact removal",
}


subtask_degra_dict = {
    v: k for k, v in degra_subtask_dict.items()
}
degradations = set(degra_subtask_dict.keys())
subtasks = set(degra_subtask_dict.values())
levels = ["very low", "low", "medium", "high", "very high"]


def generate_retrieval_embedding(state):
    #print("state: ", state)
    combining_function = state["retrieval_args"]["combining_function"]
    combiner_path = state["retrieval_args"]["combiner_path"]
    clip_model_name = state["retrieval_args"]["clip_model_name"]
    clip_model_path = state["retrieval_args"]["clip_model_path"]
    projection_dim = state["retrieval_args"]["projection_dim"]
    hidden_dim = state["retrieval_args"]["hidden_dim"]
    transform = state["retrieval_args"]["transform"]
    target_ratio = state["retrieval_args"]["target_ratio"]

    # load clip model
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # defind preprocess
    if transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(target_ratio, input_dim)
    elif transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    # load combiner model
    if combining_function.lower() == 'sum':
        if combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device)
        saved_state_dict = torch.load(combiner_path, map_location=device)
        combiner.load_state_dict(saved_state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    # generate embedding
    clip_model = clip_model.float().eval()

    text_input = clip.tokenize(["similar degradation"], context_length=77).to(device)
    with torch.no_grad():
        text_feature = clip_model.encode_text(text_input)

    image = preprocess(PIL.Image.open(state["input_img_path"])).to(device, non_blocking=True).unsqueeze(0)
    with torch.no_grad():
        image_feature = clip_model.encode_image(image)

    embedding = F.normalize(combining_function(image_feature, text_feature), dim=-1)
    print(f"generate embedding for {state['input_img_path']}, shape {embedding.shape}")

    return embedding


def retrieve_from_database(embedding, topk):
    # for now only support top1
    # connect to PostgreSQL
    try:
        conn = psycopg2.connect(dbname="agenticir_rag_test",\
                                user="postgres",\
                                host="/var/run/postgresql")
        print("Successfully connect to PostgreSQL ! ")
    except psycopg2.Error as e:
        print(f"Connection to PostgreSQL failed: {e}")

    # register with pgvector
    try:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
    except psycopg2.Error as e:
        print(f"pgvector registration failed: {e} ")

    # retrieve
    query_embedding = embedding.cpu().detach().squeeze(0).tolist()
    query = f"""
        SELECT id, name, res_seq, 1 - (embedding <=> %s::vector) AS similarity
        FROM ImgresEmbedding
        ORDER BY similarity DESC
        LIMIT {topk};
        """

    cur.execute(query, (query_embedding,))
    results = cur.fetchall()
    for _id, name, res_seq, sim in results:
        print(f"_id: {_id}, name: {name}, res_seq: {res_seq}, sim: {sim}")

    # close connection to postgresql
    cur.close()
    conn.close()

    return results # currently only support len(results)=1


def get_logger(logger_name: str,
               log_file: Optional[Path | str] = None,
               console_log_level: int = logging.INFO,
               file_log_level: int = logging.INFO,
               console_format_str: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
               file_format_str: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
               silent: bool = False
               ) -> logging.Logger:
    """Gets a logger with the specified setting.

    Args:
        logger_name (str): Name of the logger.
        log_file (Path, optional): If not None, logs to this file. Defaults to None.
        console_log_level/file_log_level (int, optional): Logging level for console/file. One of logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL. Defaults to logging.INFO.
        console_format_str/file_format_str (str, optional): Format of the log message for console/file. Defaults to '%(asctime)s - %(levelname)s - %(name)s - %(message)s'.
        silent (bool, optional): If True, does not log to console. Defaults to False.

    Returns:
        logging.Logger: Logger object.
    """

    logger_id = f"{logger_name}@{time()}"
    logger = logging.getLogger(logger_id)
    logger.setLevel(min(console_log_level, file_log_level))

    if not silent:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = ColoredFormatter(console_format_str)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(file_format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_depictqa():
    depictqa = DepictQA(
            logger=qa_logger, 
            silent=False
            )
    return depictqa
     

def get_GPT4(llm_config_path = "config.yml"):
    gpt4 = GPT4(
        config_path=llm_config_path,
        logger=qa_logger,
        silent=False,
        system_message=prompts.system_message,
        )
    return gpt4


def schedule_w_experience(state, gpt4, degradations, agenda, ps):
    def check_order(schedule: object):
        assert isinstance(schedule, dict), "Schedule should be a dict."
        assert set(schedule.keys()) == {"thought", "order"}, \
            f"Invalid keys: {schedule.keys()}."
        order = schedule["order"]
        assert set(order) == set(agenda), \
            f"{order} is not a permutation of {agenda}."
    
    with open(state["schedule_experience_path"], "r") as f:
        schedule_experience: str = json.load(f)["distilled"]
    
    schedule = gpt4(
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda,
                experience=schedule_experience
            ) + ps,
            format_check=check_order)
    
    schedule = eval(schedule)
    workflow_logger.info(f"Insights: {schedule['thought']}")
    return schedule["order"]

def reason_to_schedule(gpt4, degradations, agenda):
    insights = gpt4(
        prompt=prompts.reason_to_schedule_prompt.format(
            degradations=degradations, agenda=agenda
        ),
    )
    workflow_logger.info(f"Insights: {insights}")
    return insights

def schedule_wo_experience(state, gpt4, degradations, agenda, ps):
    insights = reason_to_schedule(gpt4, degradations, agenda)

    def check_order(order: object):
        assert isinstance(order, list), "Order should be a list."
        assert set(order) == set(agenda), f"{order} is not a permutation of {agenda}."

    order = gpt4(
        prompt=prompts.schedule_wo_retrieval_prompt.format(
            degradations=degradations, agenda=agenda, insights=insights
        ) + ps,
        format_check=check_order,
    )
    return eval(order)

def get_toolbox(state, subtask):
    # subtask: ['haze', 'low-light',...]
    if state["sim"] <= 0.9:    
        toolbox = executor.toolbox_router[subtask]
        random.shuffle(toolbox)
    # subtask: [("haze", "dehaze-former"), ...]
    else:
        toolbox = [tool for tool in executor.toolbox_router[subtask[0]] \
                    if tool.tool_name==subtask[1]]
    return toolbox

def compare_quality(depictqa, img_path1, img_path2):
    choice = depictqa(img_path=[img_path1, img_path2], task="comp_quality")
    return choice

def search_best_by_comp(candidates, state):
    best_img = candidates[0]
    
    #is_port_in_use(port, host="127.0.0.1")

    for i in range(1, len(candidates)):
        cur_img = candidates[i]
        choice = compare_quality(state["depictqa"], best_img, cur_img)
        if choice == "latter":
            best_img = cur_img
    return best_img









