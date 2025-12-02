from langgraph.graph import StateGraph, END
from copy import deepcopy as cpy
from PIL import Image, ImageOps
from pathlib import Path
import shutil
import logging
from time import localtime, strftime
from utils.util import *
from agentic import *

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, AsyncIterator
from pydantic import BaseModel
import base64
import io
import json
import asyncio


# Initialize FastAPI application
app = FastAPI(
    title="Automatic Image Restoration",
    description="use LangGraph to construct automatic image restoration workflow",
    version="0.0.1"
)


# define image restoration response
class ImageRestorationResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    steps: List[Dict[str, Any]]
    message: Optional[str] = None


# FastAPI apis, global workflow instantiation
image_graph = None

@app.on_event("startup")
async def startup_event():
    global image_graph

    image_graph = create_image_analysis_graph()
    print("LangGraph workflow has been initialized...")


@app.get("/")
async def root():
    # health check
    return {
            "status": "running",
            "service": "LangGraph Automatic Image Restoration",
            "graph_initialized": image_graph is not None
    }


@app.post("/restoration", response_model=ImageRestorationResponse)
async def image_restoration_endpoint(
#    file: UploadFile = File(...),
    file: str = "input.png",
    output_format: str = Form("PNG")
    ):
    """
    Image restoration api:
    Args:
        file: upload image file
    Returns:
        restoration path
    """
    if image_graph is None:
        raise HTTPException(status_code=500, detail="Workflow has not been initializaed")
    
    #if not file.content_type.startswith("image/"):
    #    raise HTTPException(status_code=400, detail="Please upload image")

    try:
        # read image and transfer
        # image_bytes = await file.read()
        #input_image = Image.open(io.BytesIO(image_bytes))
        input_image = Image.open(file)

        # input args
        invoke_dict = {}

        AgenticIR_dir = Path("../AgenticIR")
        CLIP4CIR_model_dir = Path("../AgenticIR/retrival_database/CLIP4CIR/models")

        # set input_img_path
        #invoke_dict["input_img_path"] = "./demo_input/001.png"
        invoke_dict["input_img_path"] = file
        invoke_dict["image"] = input_image
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
        
        final_state = image_graph.invoke(invoke_dict)
        
        # get output image
        output_image = Image.open(final_state.get("best_img_path"))
        
        if output_image is None:
            raise HTTPException(status_code=500, detail="Image restoration failed")
        
        # transform back to bytes and return
        output_bytes = pil_to_bytes(output_image, format=output_format)
        
        return Response(
            content=output_bytes,
            media_type=f"image/{output_format.lower()}",
            headers={
                "X-Output-Size": f"{output_image.width}x{output_image.height}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1146, log_level="info")



