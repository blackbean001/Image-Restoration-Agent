import sys
sys.path.append("./maxim")

import collections
import importlib
import io
import os
import base64
from io import BytesIO

from flask import Flask, request, jsonify, send_file
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from PIL import Image
import tensorflow as tf


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

port = cfg["dehazing"]["MAXIM"]["port"]
host = cfg["dehazing"]["MAXIM"]["host"]
weight_dir = cfg["dehazing"]["MAXIM"]["weight_dir"]

# app
app = Flask(__name__)

# global configuration
_MODEL_FILENAME = 'maxim'

_MODEL_VARIANT_DICT = {
    'Denoising': 'S-3',
    'Deblurring': 'S-3',
    'Deraining': 'S-2',
    'Dehazing': 'S-2',
    'Enhancement': 'S-2',
}

_MODEL_CONFIGS = {
    'variant': '',
    'dropout_rate': 0.0,
    'num_outputs': 3,
    'use_bias': True,
    'num_supervision_scales': 3,
}

_MODEL_PATH_DICT = {
    'Denoising': 'maxim_ckpt_Denoising_SIDD_checkpoint.npz',
    'Deblurring': 'maxim_ckpt_Deblurring_RealBlur_J_checkpoint.npz',
    'Deraining': 'maxim_ckpt_Deraining_Rain13k_checkpoint.npz',
    'Dehazing': 'maxim_ckpt_Dehazing_SOTS-Indoor_checkpoint.npz',
    'Enhancement': 'maxim_ckpt_Enhancement_LOL_checkpoint.npz',
}

# global models and parameters
models = {}
params_cache = {}


def recover_tree(keys, values):
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if '/' not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split('/', 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def mod_padding_symmetric(image, factor=64):
    height, width = image.shape[0], image.shape[1]
    height_pad = ((height + factor) // factor) * factor
    width_pad = ((width + factor) // factor) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = jnp.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)],
        mode='reflect')
    return image


def get_params(csdasdkpt_path):
    if ckpt_path in params_cache:
        return params_cache[ckpt_path]
    
    with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params['opt']['target']
    
    params_cache[ckpt_path] = params
    return params


def make_shape_even(image):
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = jnp.pad(image, [(0, padh), (0, padw), (0, 0)], mode='reflect')
    return image


def augment_image(image, times=8):
  """Geometric augmentation."""
  if times == 4:  # only rotate image
    images = []
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    images = np.stack(images, axis=0)
  elif times == 8:  # roate and flip image
    images = []
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    image = np.fliplr(image)
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    images = np.stack(images, axis=0)
  else:
    raise Exception(f'Error times: {times}')
  return images


def deaugment_image(images, times=8):
  """Reverse the geometric augmentation."""

  if times == 4:  # only rotate image
    image = []
    for k in range(0, 4):
      image.append(np.rot90(images[k], k=4-k))
    image = np.stack(image, axis=0)
    image = np.mean(image, axis=0)
  elif times == 8:  # roate and flip image
    image = []
    for k in range(0, 4):
      image.append(np.rot90(images[k], k=4-k))
    for k in range(0, 4):
      image.append(np.fliplr(np.rot90(images[4+k], k=4-k)))
    image = np.mean(image, axis=0)
  else:
    raise Exception(f'Error times: {times}')
  return image


def get_model(task):
    if task not in models:
        model_mod = importlib.import_module(f'maxim.models.{_MODEL_FILENAME}')
        model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
        model_configs.variant = _MODEL_VARIANT_DICT[task]
        models[task] = model_mod.Model(**model_configs)
    return models[task]


def process_image(input_img, task, geometric_ensemble=False, ensemble_times=8):
    model = get_model(task)

    ckpt_path = os.path.join(weight_dir, _MODEL_PATH_DICT[task])
    params = get_params(ckpt_path)
    
    height, width = input_img.shape[0], input_img.shape[1]
    
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]
    
    input_img = mod_padding_symmetric(input_img, factor=64)
    
    if geometric_ensemble:
        input_img = augment_image(input_img, ensemble_times)
    else:
        input_img = np.expand_dims(input_img, axis=0)
    
    preds = model.apply({'params': flax.core.freeze(params)}, input_img)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]
    
    if geometric_ensemble:
        preds = deaugment_image(preds, ensemble_times)
    else:
        preds = np.array(preds[0], np.float32)
    
    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]
    
    return preds


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input:
    - image
    - task : ['Denoising', 'Deblurring', 'Deraining', 'Dehazing', 'Enhancement']
    - geometric_ensemble (optional)
    - ensemble_times (optional)
    - return_format ['image', 'base64'] (optional)
    Return:
    - png or base64
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        if 'task' not in request.form:
            return jsonify({'error': 'Task parameter is required'}), 400
        
        image_file = request.files['image']
        task = request.form['task']
        geometric_ensemble = request.form.get('geometric_ensemble', 'false').lower() == 'true'
        ensemble_times = int(request.form.get('ensemble_times', 8))
        return_format = request.form.get('return_format', 'image')
        
        if task not in _MODEL_VARIANT_DICT:
            return jsonify({
                'error': f'Invalid task. Must be one of {list(_MODEL_VARIANT_DICT.keys())}'
            }), 400
        
        input_img = Image.open(image_file.stream).convert('RGB')
        input_img = np.asarray(input_img, np.float32) / 255.0
        
        output_img = process_image(
            input_img, 
            task, 
            ckpt_path, 
            geometric_ensemble, 
            ensemble_times
        )
        
        output_img = np.clip(output_img, 0., 1.) * 255.
        output_img = output_img.astype(np.uint8)
        output_pil = Image.fromarray(output_img)
        
        if return_format == 'base64':
            buffered = BytesIO()
            output_pil.save(buffered, format='PNG')
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return jsonify({
                'success': True,
                'image': img_str,
                'format': 'base64'
            })
        else:
            img_io = BytesIO()
            output_pil.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'task' not in request.form:
            return jsonify({'error': 'Task parameter is required'}), 400
        
        if 'ckpt_path' not in request.form:
            return jsonify({'error': 'ckpt_path parameter is required'}), 400
        
        task = request.form['task']
        ckpt_path = request.form['ckpt_path']
        geometric_ensemble = request.form.get('geometric_ensemble', 'false').lower() == 'true'
        ensemble_times = int(request.form.get('ensemble_times', 8))
        
        if task not in _MODEL_VARIANT_DICT:
            return jsonify({
                'error': f'Invalid task. Must be one of {list(_MODEL_VARIANT_DICT.keys())}'
            }), 400
        
        image_files = request.files.getlist('images')
        if not image_files:
            return jsonify({'error': 'No image files provided'}), 400
        
        results = []
        
        for idx, image_file in enumerate(image_files):
            try:
                input_img = Image.open(image_file.stream).convert('RGB')
                input_img = np.asarray(input_img, np.float32) / 255.0
                
                output_img = process_image(
                    input_img, 
                    task, 
                    ckpt_path, 
                    geometric_ensemble, 
                    ensemble_times
                )
                
                output_img = np.clip(output_img, 0., 1.) * 255.
                output_img = output_img.astype(np.uint8)
                output_pil = Image.fromarray(output_img)
                
                buffered = BytesIO()
                output_pil.save(buffered, format='PNG')
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                results.append({
                    'index': idx,
                    'filename': image_file.filename,
                    'success': True,
                    'image': img_str
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'filename': image_file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(image_files),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host=host, 
            port=port, 
            debug=False)









