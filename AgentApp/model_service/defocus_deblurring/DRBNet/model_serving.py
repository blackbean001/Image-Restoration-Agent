import os
import yaml
import sys
sys.path.append("./DRBNet")

from options.test_options import TestOptions
from datetime import datetime
import torch
import torchvision.utils as vutils
from ptflops import get_model_complexity_info
from util.util import *
from pathlib import Path
import time
import io
import lpips
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.DRBNet import DRBNet_single, DeblurNet_dual
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
from PIL import Image


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["defocus_deblurring"]["DRBNet"]["port"]
host = cfg["defocus_deblurring"]["DRBNet"]["host"]
single_ckpt = os.path.join(root_dir, cfg["defocus_deblurring"]["DRBNet"]["model_path"]["single"])
dual_ckpt = os.path.join(root_dir, cfg["defocus_deblurring"]["DRBNet"]["model_path"]["dual"])

# app
app = Flask(__name__)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global model config
class ModelConfig:
    def __init__(self, use_single=True):
        self.use_single = use_single
        self.single_model = None
        self.dual_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = ModelConfig()


def load_model():
    try:
        logger.info("Loading single image model...")
        if config.use_single:
            config.single_model = DRBNet_single().to(config.device)
            if os.path.exists(single_ckpt):
                config.single_model.load_state_dict(torch.load(single_ckpt, map_location=config.device))
                config.single_model.eval()
                logger.info("Single image model loaded successfully")
            else:
                logger.warning(f"Single model checkpoint not found: {single_ckpt}")
        else:
            logger.info("Loading dual image model...")
            config.dual_model = DeblurNet_dual().to(config.device)
            if os.path.exists(dual_ckpt):
                config.dual_model.load_state_dict(torch.load(dual_ckpt, map_location=config.device))
                config.dual_model.eval()
                logger.info("Dual image model loaded successfully")
            else:
                logger.warning(f"Dual model checkpoint not found: {dual_ckpt}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_array = np.array(img) / 255.0

    img_cropped = crop_image(img_array) * 2 - 1
    img_tensor = torch.FloatTensor(img_cropped.transpose(2, 0, 1)).unsqueeze(0)
    
    return img_tensor.to(config.device)


def postprocess_output(output_tensor):
    output_cpu = (output_tensor.cpu().numpy()[0].transpose(1, 2, 0) + 1.0) / 2.0
    output_img = (output_cpu * 255).astype(np.uint8)
    return Image.fromarray(output_img)


def tensor_to_base64(output_tensor):
    img = postprocess_output(output_tensor)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(config.device),
        'single_model_loaded': config.single_model is not None if config.use_single else "Invalid",
        'dual_model_loaded': config.dual_model is not None if not config.use_single else "Invalid"
    })


@app.route('/api/v1/deblur/single', methods=['POST'])
def deblur_single():
    try:
        if config.single_model is None:
            return jsonify({'error': 'Single model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        image_data = image_file.read()
        input_tensor = preprocess_image(image_data)
        
        start_time = time.time()
        with torch.no_grad():
            output = config.single_model(input_tensor)
        inference_time = time.time() - start_time
        
        return_format = request.form.get('format', 'base64')
        
        if return_format == 'image':
            img = postprocess_output(output)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        else:
            img_base64 = tensor_to_base64(output)
            return jsonify({
                'success': True,
                'image': img_base64,
                'inference_time': inference_time,
                'format': 'base64'
            })
            
    except Exception as e:
        logger.error(f"Error in single deblur: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/deblur/dual', methods=['POST'])
def deblur_dual():
    try:
        if config.dual_model is None:
            return jsonify({'error': 'Dual model not loaded'}), 500
        
        if 'image_c' not in request.files or 'image_r' not in request.files or 'image_l' not in request.files:
            return jsonify({'error': 'Three images (image_c, image_r, image_l) are required'}), 400
        
        image_c_data = request.files['image_c'].read()
        image_r_data = request.files['image_r'].read()
        image_l_data = request.files['image_l'].read()
        
        C = preprocess_image(image_c_data)
        R = preprocess_image(image_r_data)
        L = preprocess_image(image_l_data)
        
        start_time = time.time()
        with torch.no_grad():
            output = config.dual_model(C, R, L)
        inference_time = time.time() - start_time
        
        return_format = request.form.get('format', 'base64')
        
        if return_format == 'image':
            img = postprocess_output(output)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        else:
            img_base64 = tensor_to_base64(output)
            return jsonify({
                'success': True,
                'image': img_base64,
                'inference_time': inference_time,
                'format': 'base64'
            })
            
    except Exception as e:
        logger.error(f"Error in dual deblur: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'DRBNet Defocus Deblurring Service',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/api/v1/deblur/single': 'POST - Single image deblurring',
            '/api/v1/deblur/dual': 'POST - Dual image deblurring',
        },
        'usage': {
            'single': 'POST image file as "image", optional form param "format" (base64 or image)',
            'dual': 'POST three image files as "image_c", "image_r", "image_l"',
        }
    })


if __name__ == '__main__':
    logger.info("Starting DRBNet Inference Service...")
    load_model()
    
    app.run(host=host, port=port, debug=False)













