import sys
sys.path.append("./SwinIR")

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import torch
import yaml
import io
import base64
from PIL import Image
from models.network_swinir import SwinIR as net
from werkzeug.utils import secure_filename


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["denoising"]["SwinIR"]["port"]
host = cfg["denoising"]["SwinIR"]["host"]
model_path = os.path.join(root_dir, cfg["denoising"]["SwinIR"]["model_path"])


# app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def define_model(task, scale=1, noise=15, jpeg=40, large_model=False):
    if task == 'classical_sr':
        model = net(upscale=scale, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    elif task == 'lightweight_sr':
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
                    num_heads=[6, 6, 6, 6], mlp_ratio=2,
                    upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif task == 'real_sr':
        if not large_model:
            model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                        upsampler='nearest+conv', resi_connection='1conv')
        else:
            model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim=240, num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    elif task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    elif task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    elif task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    elif task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    else:
        raise ValueError(f"Unknown task: {task}")

    return model, param_key_g


def load_model(task, scale=1, noise=15, jpeg=40, large_model=False):
    model_key = f"{task}_{scale}_{noise}_{jpeg}_{large_model}"
    
    if model_key not in models:
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        
        try:
            model, param_key_g = define_model(task, scale, noise, jpeg, large_model)
            pretrained_model = torch.load(model_path, map_location=device)
            model.load_state_dict(
                pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() 
                else pretrained_model, strict=True
            )
            model.eval()
            model = model.to(device)
            models[model_key] = model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, f"Error loading model: {str(e)}"
    
    print("Load model success: ", model_path)

    return models[model_key], None


def process_image(img_lq, model, scale, window_size, tile=None, tile_overlap=32):

    img_lq = np.transpose(
        img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], 
        (2, 0, 1)
    )
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
    
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    
    with torch.no_grad():
        if tile is None:
            output = model(img_lq)
        else:
            b, c, h, w = img_lq.size()
            tile_size = min(tile, h, w)
            stride = tile_size - tile_overlap
            h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
            w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
            E = torch.zeros(b, c, h * scale, w * scale).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx + tile_size, w_idx:w_idx + tile_size]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[..., h_idx * scale:(h_idx + tile_size) * scale, 
                      w_idx * scale:(w_idx + tile_size) * scale].add_(out_patch)
                    W[..., h_idx * scale:(h_idx + tile_size) * scale, 
                      w_idx * scale:(w_idx + tile_size) * scale].add_(out_patch_mask)
            output = E.div_(W)

        output = output[..., :h_old * scale, :w_old * scale]
    
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'loaded_models': len(models)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input:
        - image
        - task (real_sr, color_dn, color_jpeg_car)
        - scale
        - noise
        - jpeg
        - large_model
        - tile
        - tile_overlap
        - return_type (file/base64)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        task = request.form.get('task', 'real_sr')
        scale = int(request.form.get('scale', 1))
        noise = int(request.form.get('noise', 15))
        jpeg = int(request.form.get('jpeg', 40))
        large_model = request.form.get('large_model', 'false').lower() == 'true'
        tile = request.form.get('tile', None)
        tile = int(tile) if tile else None
        tile_overlap = int(request.form.get('tile_overlap', 32))
        return_type = request.form.get('return_type', 'file')
        
        if task not in ['real_sr', 'color_dn', 'color_jpeg_car']:
            return jsonify({'error': 'Invalid task type'}), 400
        
        window_size = 7 if task in ['jpeg_car', 'color_jpeg_car'] else 8
        
        model, error = load_model(task, scale, noise, jpeg, large_model)
        if error:
            return jsonify({'error': error}), 500
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_lq = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_lq is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        img_lq = img_lq.astype(np.float32) / 255.
        
        output = process_image(img_lq, model, scale, window_size, tile, tile_overlap)
        
        print("return_type: ", return_type)

        if return_type == 'base64':
            _, buffer = cv2.imencode('.png', output)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True,
                'image': img_base64,
                'format': 'png'
            })
        else:
            output_filename = secure_filename(f"output_{file.filename}")
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, output)
            print("Save image to: ", output_path)
            return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("error: ", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'Empty file list'}), 400
        
        task = request.form.get('task', 'real_sr')
        scale = int(request.form.get('scale', 1))
        noise = int(request.form.get('noise', 15))
        jpeg = int(request.form.get('jpeg', 40))
        large_model = request.form.get('large_model', 'false').lower() == 'true'
        tile = request.form.get('tile', None)
        tile = int(tile) if tile else None
        tile_overlap = int(request.form.get('tile_overlap', 32))
        
        window_size = 7 if task in ['jpeg_car', 'color_jpeg_car'] else 8
        
        model, error = load_model(task, scale, noise, jpeg, large_model)
        if error:
            return jsonify({'error': error}), 500
        
        results = []
        for file in files:
            try:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_lq = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_lq is None:
                    results.append({'filename': file.filename, 'error': 'Failed to decode image'})
                    continue
                
                img_lq = img_lq.astype(np.float32) / 255.
                
                output = process_image(img_lq, model, scale, window_size, tile, tile_overlap)
                
                _, buffer = cv2.imencode('.png', output)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'image': img_base64
                })
            except Exception as e:
                results.append({'filename': file.filename, 'error': str(e)})
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host=host, port=port, debug=False)























