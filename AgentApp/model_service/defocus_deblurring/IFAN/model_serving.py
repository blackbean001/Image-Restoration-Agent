from flask import Flask, request, jsonify, send_file

import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import json
import time
import os
import sys
sys.path.append("./IFAN")

import numpy as np
from pathlib import Path
import gc
import math
import random
import traceback

from utils import *
from models import create_model
import models.archs.LPIPS as LPIPS

from data_loader.utils import load_file_list, read_frame, refine_image
from ckpt_manager import CKPT_Manager


app = Flask(__name__)

# save global model and config
model_network = None
model_config = None
device = None

# load config
def load_model_configs(config_path="../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(config):
    """load_model"""
    global model_network, model_config, device
    
    model_config = config
    device = torch.device(config.device if config.cuda else 'cpu')
    
    model = create_model(config)
    network = model.get_network().eval()
    
    ckpt_manager = CKPT_Manager(
        config.LOG_DIR.ckpt, 
        config.mode, 
        config.cuda, 
        config.max_ckpt_num
    )
    
    load_state, ckpt_name = ckpt_manager.load_ckpt(
        network,
        by_score=config.EVAL.load_ckpt_by_score,
        name=config.EVAL.ckpt_name,
        abs_name=ckpt_abs,
        epoch=config.EVAL.ckpt_epoch
    )
    
    print(f'Loading checkpoint "{ckpt_name}" on model "{config.mode}": {load_state}')
    
    model_network = network
    return network, ckpt_name


def preprocess_image(image_bytes, refine_val=4, rotate=None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if rotate is not None:
        img = cv2.rotate(img, rotate)
    
    img = img.astype(np.float32) / 255.0
    
    img = refine_image(np.expand_dims(img, axis=0), refine_val)
    
    img_tensor = torch.FloatTensor(img.transpose(0, 3, 1, 2).copy()).to(device)
    
    return img_tensor


def postprocess_image(output_tensor):
    output_np = output_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    
    output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
    
    output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return output_np


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_network is not None
    })


@app.route('/deblur', methods=['POST'])
def deblur_image():
    try:
        if model_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        image_bytes = file.read()
        
        norm_val = float(request.form.get('norm_val', model_config.norm_val))
        refine_val = int(request.form.get('refine_val', model_config.refine_val))
        output_format = request.form.get('format', 'png')  # png or jpg
        
        input_tensor = preprocess_image(image_bytes, norm_val, refine_val)
        
        with torch.no_grad():
            out = model_network(input_tensor, is_train=False)
            
            if model_config.cuda:
                torch.cuda.synchronize()
            
            output_tensor = out['result']
        
        output_np = postprocess_image(output_tensor)
        
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            _, buffer = cv2.imencode('.jpg', output_np, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime_type = 'image/jpeg'
        else:
            _, buffer = cv2.imencode('.png', output_np)
            mime_type = 'image/png'
        
        output_bytes = io.BytesIO(buffer.tobytes())
        output_bytes.seek(0)
        
        return send_file(
            output_bytes,
            mimetype=mime_type,
            as_attachment=True,
            download_name=f'deblurred.{output_format}'
        )
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/deblur_dual', methods=['POST'])
def deblur_image_dual():
    try:
        if model_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'center' not in request.files or 'left' not in request.files or 'right' not in request.files:
            return jsonify({'error': 'Need center, left, and right images'}), 400
        
        center_bytes = request.files['center'].read()
        left_bytes = request.files['left'].read()
        right_bytes = request.files['right'].read()
        
        norm_val = float(request.form.get('norm_val', model_config.norm_val))
        refine_val = int(request.form.get('refine_val', model_config.refine_val))
        output_format = request.form.get('format', 'png')
        
        C = preprocess_image(center_bytes, norm_val, refine_val)
        L = preprocess_image(left_bytes, norm_val, refine_val)
        R = preprocess_image(right_bytes, norm_val, refine_val)
        
        with torch.no_grad():
            out = model_network(C, R, L, is_train=False)
            
            if model_config.cuda:
                torch.cuda.synchronize()
            
            output_tensor = out['result']
        
        output_np = postprocess_image(output_tensor)
        
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            _, buffer = cv2.imencode('.jpg', output_np, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime_type = 'image/jpeg'
        else:
            _, buffer = cv2.imencode('.png', output_np)
            mime_type = 'image/png'
        
        output_bytes = io.BytesIO(buffer.tobytes())
        output_bytes.seek(0)
        
        return send_file(
            output_bytes,
            mimetype=mime_type,
            as_attachment=True,
            download_name=f'deblurred.{output_format}'
        )

    except Exception as e:
        print(f"Error processing images: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def initialize_service(config):
    print("Initializing deblur service...")
    load_model(config)
    print("Model loaded successfully!")


if __name__ == "__main__":
    cfg = load_model_configs()

    port = cfg["defocus_deblurring"]["IFAN"]["port"]
    host = cfg["defocus_deblurring"]["IFAN"]["host"]
    ckpt_name = cfg["defocus_deblurring"]["IFAN"]["ckpt_name"]
    ckpt_abs = cfg["defocus_deblurring"]["IFAN"]["ckpt_abs"]
    config_name = cfg["defocus_deblurring"]["IFAN"]["config"]
    mode = cfg["defocus_deblurring"]["IFAN"]["mode"]
    project = cfg["defocus_deblurring"]["IFAN"]["project"]

    config_lib = importlib.import_module(f'configs.{config_name}')
    config = config_lib.get_config(project, mode, config_name)
    config.is_train = False
    config.cuda = torch.cuda.is_available()
    config.device = 'cuda' if config.cuda else 'cpu'
    
    config.EVAL.ckpt_name = args.ckpt_name
    config.EVAL.ckpt_abs_name = args.ckpt_abs
    config.EVAL.load_ckpt_by_score = False
    
    initialize_service(config)
    
    app.run(host=host, port=port, debug=False)















