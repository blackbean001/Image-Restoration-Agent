from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from enum import Enum
from typing import Optional
import uvicorn
import yaml

def load_model_configs(config_path="../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()
port = cfg["models"]["brightening"]["port"]
host = cfg["models"]["brightening"]["host"]


app = FastAPI(
    title="image_brightening_service",
    description="provide multiple image brightening service API",
    version="1.0.0"
)


class BrighteningMethod(str, Enum):
    constant_shift = "constant_shift"
    gamma_correction = "gamma_correction"
    histogram_equalization = "histogram_equalization"


class ImageBrightener:
    """
    base class for image brightening
    """
    @staticmethod
    def process_image(img: np.ndarray, method: BrighteningMethod) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        if method == BrighteningMethod.constant_shift:
            v = ImageBrightener._constant_shift(v)
        elif method == BrighteningMethod.gamma_correction:
            v = ImageBrightener._gamma_correction(v)
        elif method == BrighteningMethod.histogram_equalization:
            v = ImageBrightener._histogram_equalization(v)
        
        hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result

    @staticmethod
    def _constant_shift(v: np.ndarray, shift: int = 40) -> np.ndarray:
        img = np.clip(np.uint16(v) + shift, 0, 255)
        return img.round().astype(np.uint8)

    @staticmethod
    def _gamma_correction(v: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        img = cv2.pow(v / 255.0, 1.0 / gamma) * 255
        return img.clip(0, 255).round().astype(np.uint8)

    @staticmethod
    def _histogram_equalization(v: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(v)


@app.get("/")
async def root():
    """API root"""
    return {
            "message": "image brightening API",
            "version": 1.0.0,
            "endpoints": {
                "brighten": "/brighten",
                "methods": "/methods",
                "health": "/health"
                }
           }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/methods")
async def get_methods():
    return {
        "methods": [
            {
                "name": "constant_shift",
                "description": "constant shift - add fix value on channel v",
                "parameters": {"shift": 40}
            },
            {
                "name": "gamma_correction",
                "description": "gamma correction - use power function to correct brightness",
                "parameters": {"gamma": 1.5}
            },
            {
                "name": "histogram_equalization",
                "description": "adaptive histogram equalization - CLAHE algorithm",
                "parameters": {"clipLimit": 2.0, "tileGridSize": (8, 8)}
            }
        ]
    }


@app.get("/brighten")
async def brighten_image(
    file: UploadFile = File(..., description="image to be processed"),
    method: BrighteningMethod = Query(
        BrighteningMethod.histogram_equalization,
        description="brightening methods"
    ),
    output_format: str = Query(
        "png",
        description="output format (png, jpg, jpeg)",
        regex="^(png|jpg|jpeg)$"
    )
):
    try:
        contents = await file.read()
        
        img = bytes_to_image(contents)
        
        result_img = ImageBrightener.process_image(img, method)
        
        output_format_ext = f".{output_format}"
        result_bytes = image_to_bytes(result_img, output_format_ext)
        
        media_type = f"image/{output_format}"
        return StreamingResponse(
            BytesIO(result_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=brightened.{output_format}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed processing image: {str(e)}")


@app.post("/brighten/batch")
async def brighten_batch(
    method: BrighteningMethod = Query(
        BrighteningMethod.histogram_equalization,
        description="brightening method"
    )
):
    return {
        "message": "batch processing (tbd)"
        "method": method
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )












