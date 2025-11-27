import os
import os.path as osp
import logging
import torch
import yaml
import argparse
import random
from collections import OrderedDict
from flask import Flask, request, jsonify
import tempfile
import shutil
from threading import Lock

import hat.archs
import hat.data
import hat.models
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, ordered_yaml, _postprocess_yml_value
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils import set_random_seed
from basicsr.models.sr_model import SRModel


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

root_dir = cfg["root_dir"]
port = cfg["super_resolution"]["HAT"]["port"]
host = cfg["super_resolution"]["HAT"]["host"]
opt_config = os.path.join(root_dir, cfg["super_resolution"]["HAT"]["config"])
print("opt_config: ", opt_config)

ckpt_path = os.path.join(root_dir, cfg["super_resolution"]["HAT"]["model_path"])

# app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global model cache
model_cache = {}
model_lock = Lock()


def custom_parse_options_from_config(config_path, root_path, is_train=False):
    """Parse options from config file"""
    with open(config_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    opt['dist'] = False
    opt['rank'], opt['world_size'] = 0, 1

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['is_train'] = is_train

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if not is_train:
        #results_root = osp.join(opt['path'].get('results', root_path), opt['name'])
        results_root = "temp_output"
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        #opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['visualization'] = results_root
        opt['path']['pretrain_network'] = ckpt_path
    print("opt: ", opt)
    return opt


def load_model(config_path, root_path):
    """Load or retrieve cached model"""
    with model_lock:
        if config_path not in model_cache:
            opt = custom_parse_options_from_config(config_path, root_path, is_train=False)
            torch.backends.cudnn.benchmark = True

            # Create directories
            make_exp_dirs(opt)

            # Build model
            model = build_model(opt)
            model_cache[config_path] = {'model': model, 'opt': opt}

        return model_cache[config_path]['model'], model_cache[config_path]['opt']

# load_model globally
MODEL, OPT = load_model(opt_config, "./")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })


@app.route('/super_resolution', methods=['POST'])
def super_resolution():
    """
    Super resolution endpoint
    Expected JSON data:
    - input_path: path to input image file
    - output_path: path to save output image
    """
    temp_input_dir = None
    temp_output_dir = None
    
    try:
        data = request.get_json()
        
        if not data or 'input_path' not in data or 'output_path' not in data:
            return jsonify({'error': 'input_path and output_path are required'}), 400

        input_path = data['input_path']
        output_path = data['output_path']

        if not osp.exists(input_path):
            return jsonify({'error': f'Input file not found: {input_path}'}), 404

        output_dir = osp.dirname(output_path)
        if output_dir and not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        temp_input_dir = "temp_input"
        temp_output_dir = "temp_output"
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        temp_input_file = osp.join(temp_input_dir, osp.basename(input_path))
        shutil.copy2(input_path, temp_input_file)

        temp_opt = OPT.copy()
        temp_opt['path']['visualization'] = temp_output_dir
        
        test_loaders = []
        for dataset_name, dataset_opt in sorted(temp_opt['datasets'].items()):
            dataset_opt['dataroot_lq'] = temp_input_dir
            
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=temp_opt['num_gpu'],
                dist=temp_opt['dist'], sampler=None, seed=temp_opt['manual_seed'])
            test_loaders.append(test_loader)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            MODEL.validation(test_loader, current_iter=temp_opt['name'],
                           tb_logger=None, save_img=True)
        
        print("temp_output_dir: ", temp_output_dir)
        output_found = False
        for root, dirs, files in os.walk(temp_output_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    generated_file = osp.join(root, file)
                    print("generated_file: ", generated_file)
                    print("output_path: ", output_path)
                    shutil.copy2(generated_file, output_path)
                    output_found = True
                    break
            if output_found:
                break

        #if not output_found:
        #    return jsonify({'error': 'No output image generated'}), 500

        return jsonify({
            'status': 'success',
            'input_path': input_path,
            'output_path': output_path,
            'message': 'Super resolution completed successfully'
        })

    except Exception as e:
        logging.error(f"Error during super resolution: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    finally:
        if temp_input_dir and osp.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir, ignore_errors=True)
        if temp_output_dir and osp.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the model cache"""
    with model_lock:
        model_cache.clear()
        torch.cuda.empty_cache()
    return jsonify({'status': 'cache cleared'})

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run Flask app
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )


















