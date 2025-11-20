import sys
sys.path.append("./DiffBIR")

from flask import Flask, request, jsonify, send_file
from typing import List, Tuple, Optional
import os
import math
import io
import base64
import yaml
from werkzeug.utils import secure_filename

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from model.cond_fn import MSEGuidance
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["super_resolution"]["DiffBIR"]["port"]
host = cfg["super_resolution"]["DiffBIR"]["host"]
opt_config = os.path.join(root_dir, cfg["super_resolution"]["DiffBIR"]["config"])
ckpt_path = os.path.join(root_dir, cfg["super_resolution"]["DiffBIR"]["model_path"])
swinir_ckpt_path = os.path.join(root_dir, cfg["super_resolution"]["DiffBIR"]["swinir_ckpt_path"])


# app
app = Flask(__name__)

CONFIG = {
    'model': None,
    'device': 'cuda',
    'initialized': False
}


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_device(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"
    else:
        disable_xformers()
        if device == "mps":
            if not torch.backends.mps.is_available():
                print("MPS not available, using CPU")
                device = "cpu"
    print(f'Using device: {device}')
    return device


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13

    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if not tiled:
        samples = sampler.sample(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]

    return preds, stage1_preds


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_initialized': CONFIG['initialized'],
        'device': CONFIG['device']
    })


@app.route('/initialize', methods=['POST'])
def initialize_model():
    try:
        data = request.json
        device = data.get('device', 'cuda')
        reload_swinir = data.get('reload_swinir', False)
        seed = data.get('seed', 231)

        pl.seed_everything(seed)

        device = check_device(device)
        CONFIG['device'] = device

        model: ControlLDM = instantiate_from_config(OmegaConf.load(opt_config))
        load_state_dict(model, torch.load(ckpt_path, map_location="cpu"), strict=True)

        if reload_swinir:
            if not hasattr(model, "preprocess_model"):
                return jsonify({'error': 'Model does not have a preprocess model'}), 400
            print(f"Reloading SwinIR model from {ckpt_path}")
            load_state_dict(model.preprocess_model, torch.load(ckpt_path, map_location="cpu"), strict=True)

        model.freeze()
        model.to(device)

        CONFIG['model'] = model
        CONFIG['initialized'] = True

        return jsonify({
            'status': 'success',
            'message': 'Model initialized successfully',
            'device': device
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inference', methods=['POST'])
def inference():
    if not CONFIG['initialized']:
        return jsonify({'error': 'Model not initialized. Call /initialize first'}), 400

    try:
        steps = int(request.form.get('steps', 50))
        sr_scale = float(request.form.get('sr_scale', 1.0))
        color_fix_type = request.form.get('color_fix_type', 'wavelet')
        disable_preprocess = request.form.get('disable_preprocess_model', 'false').lower() == 'true'
        tiled = request.form.get('tiled', 'false').lower() == 'true'
        tile_size = int(request.form.get('tile_size', 512))
        tile_stride = int(request.form.get('tile_stride', 256))
        
        use_guidance = request.form.get('use_guidance', 'false').lower() == 'true'
        g_scale = float(request.form.get('g_scale', 0.0))
        g_t_start = int(request.form.get('g_t_start', 1001))
        g_t_stop = int(request.form.get('g_t_stop', -1))
        g_space = request.form.get('g_space', 'latent')
        g_repeat = int(request.form.get('g_repeat', 5))

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        lq = Image.open(file.stream).convert("RGB")
        
        if sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        
        if not tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, tile_size)
        
        x = pad(np.array(lq_resized), scale=64)

        cond_fn = None
        if use_guidance:
            cond_fn = MSEGuidance(
                scale=g_scale, t_start=g_t_start, t_stop=g_t_stop,
                space=g_space, repeat=g_repeat
            )

        preds, stage1_preds = process(
            CONFIG['model'], [x], steps=steps,
            strength=1.0,
            color_fix_type=color_fix_type,
            disable_preprocess_model=disable_preprocess,
            cond_fn=cond_fn,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )

        pred = preds[0]
        
        pred = pred[:lq_resized.height, :lq_resized.width, :]
        
        result_img = Image.fromarray(pred).resize(lq.size, Image.LANCZOS)

        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'status': 'success',
            'image': img_str,
            'original_size': lq.size,
            'processed_size': (pred.shape[1], pred.shape[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inference_file', methods=['POST'])
def inference_file():
    if not CONFIG['initialized']:
        return jsonify({'error': 'Model not initialized. Call /initialize first'}), 400

    try:
        steps = int(request.form.get('steps', 50))
        sr_scale = float(request.form.get('sr_scale', 1.0))
        color_fix_type = request.form.get('color_fix_type', 'wavelet')
        disable_preprocess = request.form.get('disable_preprocess_model', 'false').lower() == 'true'
        tiled = request.form.get('tiled', 'false').lower() == 'true'
        tile_size = int(request.form.get('tile_size', 512))
        tile_stride = int(request.form.get('tile_stride', 256))

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        lq = Image.open(file.stream).convert("RGB")
        
        if sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        
        if not tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, tile_size)
        
        x = pad(np.array(lq_resized), scale=64)

        preds, _ = process(
            CONFIG['model'], [x], steps=steps,
            strength=1.0,
            color_fix_type=color_fix_type,
            disable_preprocess_model=disable_preprocess,
            cond_fn=None,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )

        pred = preds[0]
        pred = pred[:lq_resized.height, :lq_resized.width, :]
        result_img = Image.fromarray(pred).resize(lq.size, Image.LANCZOS)

        img_io = io.BytesIO()
        result_img.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='result.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host=host, 
            port=port, 
            debug=False)
















