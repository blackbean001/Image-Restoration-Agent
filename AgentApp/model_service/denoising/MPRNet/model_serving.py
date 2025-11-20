import sys
sys.path.append("MPRNet")

from flask import Flask, request, send_file, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import yaml
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import io
import uuid
from datetime import datetime


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["denoising"]["MPRNet"]["port"]
host = cfg["denoising"]["MPRNet"]["host"]
weight_dir = os.path.join(root_dir, cfg["denoising"]["MPRNet"]["weight_dir"])

# app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

models_cache = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


def get_model(task):
    if task in models_cache:
        return models_cache[task]
    
    load_file = run_path(os.path.join("MPRNet", task, "MPRNet.py"))
    model = load_file['MPRNet']()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    weights = os.path.join(weight_dir, f"model_{task.lower()}.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at {weights}")
    
    model = load_checkpoint(model, weights)
    model.eval()
    
    models_cache[task] = model

    return model


def process_image(image_path, task, img_multiple_of=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(task)
    
    img = Image.open(image_path).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).to(device)
    
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)
    
    restored = restored[0]
    restored = torch.clamp(restored, 0, 1)
    
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    return restored


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'loaded_models': list(models_cache.keys())
    })


@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        task = request.form.get('task', 'Denoising')
        if task not in ['Deblurring', 'Denoising', 'Deraining']:
            return jsonify({'error': 'Invalid task. Choose from: Deblurring, Denoising, Deraining'}), 400
        
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        file.save(input_path)
        
        restored = process_image(input_path, task)
        
        output_filename = f"{unique_id}_restored.png"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        
        os.remove(input_path)
        
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=f'restored_{filename}')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_base64', methods=['POST'])
def process_base64():
    import base64
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        task = data.get('task', 'Denoising')
        if task not in ['Deblurring', 'Denoising', 'Deraining']:
            return jsonify({'error': 'Invalid task'}), 400
        
        image_data = base64.b64decode(data['image'])
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.png")
        
        with open(input_path, 'wb') as f:
            f.write(image_data)
        
        restored = process_image(input_path, task)
        
        output_path = os.path.join(app.config['RESULT_FOLDER'], f"{unique_id}_restored.png")
        cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        
        with open(output_path, 'rb') as f:
            restored_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(input_path)
        os.remove(output_path)
        
        return jsonify({
            'success': True,
            'image': restored_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify({
        'tasks': ['Deblurring', 'Denoising', 'Deraining']
    })


if __name__ == '__main__':
    app.run(host=host,
            port=port,
            debug=False)


