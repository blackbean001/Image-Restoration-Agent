import sys
sys.path.append("RIDCP_dehazing")

from flask import Flask, request, jsonify, send_file
import cv2
import os
import yaml
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from basicsr.utils import img2tensor, tensor2img
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet


app = Flask(__name__)

# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["dehazing"]["RIDCP"]["port"]
host = cfg["dehazing"]["RIDCP"]["host"]
WEIGHT_PATH = os.path.join(root_dir, cfg["dehazing"]["RIDCP"]["weight_path"])

MAX_SIZE = 1500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# global model variable
model = None

def load_model(weight_path, use_weight=False, alpha=1.0):
    """load dehazing model"""
    global model

    model = VQWeightDehazeNet(
        codebook_params=[[64, 1024, 512]],
        LQ_stage=True,
        use_weight=use_weight,
        weight_alpha=alpha
    ).to(DEVICE)

    model.load_state_dict(torch.load(WEIGHT_PATH)['params'], strict=False)
    model.eval()
    print("Load model success")

    return True


def process_image(img, max_size=1500):
    if img.max() > 255.0:
        img = img / 255.0
    if img.shape[-1] > 3:
        img = img[:, :, :3]

    img_tensor = img2tensor(img).to(DEVICE) / 255.
    img_tensor = img_tensor.unsqueeze(0)

    max_size_pixels = max_size ** 2
    h, w = img_tensor.shape[2:]

    with torch.no_grad():
        if h * w < max_size_pixels:
            output, _ = model.test(img_tensor)
        else:
            down_img = torch.nn.functional.interpolate(
                img_tensor, size=(h//2, w//2), mode='bilinear', align_corners=False
            )
            output, _ = model.test(down_img)
            output = torch.nn.functional.interpolate(
                output, size=(h, w), mode='bilinear', align_corners=False
            )


    output_img = tensor2img(output)
    return output_img


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })


@app.route('/dehaze', methods=['POST'])
def dehaze_image():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please call /load_model first.'
        }), 400
    
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image file provided'
        }), 400
    
    try:
        file = request.files['image']
        max_size = int(request.form.get('max_size', MAX_SIZE))
        return_type = request.form.get('return_type', 'file')  # 'file' or 'base64'
        
        img_bytes = file.read()

        #img = Image.open(io.BytesIO(img_bytes))

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image file'
            }), 400
        
        output_img = process_image(img, max_size)

        if return_type == 'base64':
            _, buffer = cv2.imencode('.png', output_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'status': 'success',
                'image': img_base64,
                'format': 'base64'
            })
        else:
            _, buffer = cv2.imencode('.png', output_img)
            return send_file(
                BytesIO(buffer),
                mimetype='image/png',
                as_attachment=True,
                download_name='dehazed_image.png'
            )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/dehaze_base64', methods=['POST'])
def dehaze_base64():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please call /load_model first.'
        }), 400
    
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        max_size = data.get('max_size', MAX_SIZE)
        
        if not img_base64:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image data'
            }), 400
        
        output_img = process_image(img, max_size)
        
        _, buffer = cv2.imencode('.png', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': img_base64,
            'format': 'base64'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    load_model(WEIGHT_PATH, True)

    app.run(host=host,
        port=port,
        debug=False
        )

















