import sys
sys.path.append("./FBCNN")

import os
import logging
import numpy as np
import torch
import yaml
import cv2
from flask import Flask, request, jsonify
import base64
from utils import utils_image as util

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["jpeg_compression_artifact_removal"]["FBCNN"]["port"]
host = cfg["jpeg_compression_artifact_removal"]["FBCNN"]["host"]
weight_dir = os.path.join(root_dir, cfg["jpeg_compression_artifact_removal"]["FBCNN"]["weight_dir"])

# global params
model = None
device = None
n_channels = 3

def load_model():
    global model, device

    model_name = 'fbcnn_color.pth'
    model_path = os.path.join(weight_dir, model_name)

    if not os.path.exists(model_path):
        print(f'Downloading model to {model_path}')
        import requests
        url = f'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{model_name}'
        r = requests.get(url, allow_redirects=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        print('Model downloaded successfully')
    else:
        print(f'Model already exists at {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    from models.network_fbcnn import FBCNN as net
    nc = [64, 128, 256, 512]
    nb = 4

    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)
    print('Model loaded successfully')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized'
    })

@app.route('/denoise', methods=['POST'])
def denoise_image():
    try:
        data = request.get_json()
        
        if not data or 'input_path' not in data or 'output_path' not in data:
            return jsonify({'error': 'input_path and output_path are required'}), 400

        input_path = data['input_path']
        output_path = data['output_path']
        qf = data.get('qf', 'blind')

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            return jsonify({'error': f'Input file not found: {input_path}'}), 404

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 读取图片
        img_L = util.imread_uint(input_path, n_channels=n_channels)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # 处理图片
        with torch.no_grad():
            if qf == 'blind':
                img_E, QF = model(img_L)
                QF = 1 - QF
                predicted_qf = round(float(QF * 100))
            else:
                try:
                    QF_set = int(qf)
                    if QF_set < 0 or QF_set > 100:
                        return jsonify({'error': 'QF must be between 0 and 100'}), 400

                    qf_input = torch.tensor([[1 - QF_set / 100]]).to(device)
                    img_E, QF = model(img_L, qf_input)
                    QF = 1 - QF
                    predicted_qf = QF_set
                except ValueError:
                    return jsonify({'error': 'Invalid QF value. Must be "blind" or an integer'}), 400

        # 保存结果
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)
        util.imsave(img_E, output_path)

        return jsonify({
            'success': True,
            'predicted_qf': predicted_qf,
            'input_path': input_path,
            'output_path': output_path
        })

    except Exception as e:
        logging.error(f'Error processing image: {str(e)}')
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/denoise_base64', methods=['POST'])
def denoise_image_base64():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = base64.b64decode(data['image'])
        qf = data.get('qf', 'blind')

        nparr = np.frombuffer(image_data, np.uint8)
        img_L = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_L is None:
            return jsonify({'error': 'Invalid image data'}), 400

        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        with torch.no_grad():
            if qf == 'blind':
                img_E, QF = model(img_L)
                QF = 1 - QF
                predicted_qf = round(float(QF * 100))
            else:
                QF_set = int(qf)
                qf_input = torch.tensor([[1 - QF_set / 100]]).to(device)
                img_E, QF = model(img_L, qf_input)
                QF = 1 - QF
                predicted_qf = QF_set

        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)

        _, buffer = cv2.imencode('.png', img_E)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'predicted_qf': predicted_qf,
            'image': img_base64
        })

    except Exception as e:
        logging.error(f'Error processing image: {str(e)}')
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(host=host, port=port, debug=False)

