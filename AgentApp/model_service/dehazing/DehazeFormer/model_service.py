import sys
sys.path.append("DehazeFormer")

import os
import io
import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, send_file, jsonify
from collections import OrderedDict
from PIL import Image

from utils import AverageMeter, write_img, chw_to_hwc, hwc_to_chw
from datasets.loader import PairLoader, SingleLoader

from models import *


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()
port = cfg["dehazing"]["Dehazeformer"]["port"]
host = cfg["dehazing"]["Dehazeformer"]["host"]
model_name = cfg["dehazing"]["Dehazeformer"]["model_name"]
model_path = cfg["dehazing"]["Dehazeformer"]["model_path"]

# app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


model = None
device = None


def read_img_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = np.array(img).astype('float32') / 255.0
    return img


def write_img_to_bytes(img):
    img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    is_success, buffer = cv2.imencode(".png", img)
    if is_success:
        return io.BytesIO(buffer)
    return None


def load_model_weights(save_dir):
    state_dict = torch.load(save_dir, map_location=device)['state_dict']
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    return new_state_dict


def process_image(img_bytes):
    img = read_img_from_bytes(img_bytes)
    
    img = img * 2 - 1
    
    img_chw = hwc_to_chw(img)

    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    model.eval()

    with torch.no_grad():
        output = model(img_tensor).clamp_(-1, 1)
        output = output * 0.5 + 0.5
    
    out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())

    return out_img


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/dehaze', methods=['POST'])
def dehaze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        img_bytes = file.read()

        output_img = process_image(img_bytes)

        output_bytes = write_img_to_bytes(output_img)
        if output_bytes is None:
            return jsonify({'error': 'Failed to encode output image'}), 500
        
        return send_file(
            output_bytes,
            mimetype='image/png',
            as_attachment=True,
            download_name='dehazed_output.png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_dehaze', methods=['POST'])
def batch_dehaze():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            img_bytes = file.read()
            output_img = process_image(img_bytes)
            output_bytes = write_img_to_bytes(output_img)
            
            if output_bytes:
                results.append({
                    'filename': file.filename,
                    'status': 'success'
                })
        
        return jsonify({
            'message': f'Processed {len(results)} images successfully',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_model(model_name='dehazeformer-b', model_path='/App/AgentApp/weights/dehazing/DehazeFormer/saved_models/indoor/dehazeformer-b.pth'):
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = eval(model_name.replace('-', '_'))()
    model.to(device)
    
    if os.path.exists(model_path):
        print(f'Loading model from: {model_path}')
        model.load_state_dict(load_model_weights(model_path))
        print('Model loaded successfully!')
    else:
        print(f'Warning: Model file not found at {model_path}')
        print('Please ensure the model file exists before making requests.')
    
    model.eval()


if __name__ == "__main__":

    initialize_model(model_name, model_path)

    # start Flask service
    print(f'Starting Flask server on {host}:{port}')
    app.run(host=host,
            port=port,
            debug=False)


