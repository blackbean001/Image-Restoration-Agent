import os
import os.path as osp
import logging
import torch
import yaml
import argparse
import random
from collections import OrderedDict
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
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
ckpt_path = os.path.join(root_dir, cfg["super_resolution"]["HAT"]["model_path"])


# app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()


# Global model cache
model_cache = {}
model_lock = Lock()

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        results_root = osp.join(opt['path'].get('results', root_path), opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['network_g'] = ckpt_path
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


@app.route('/test', methods=['POST'])
def test_image():
    """
    Test endpoint for image super-resolution
    Expected form data:
    - config: path to YAML config file
    - image: image file to process (optional if using dataset from config)
    - save_img: whether to save output image (true/false)
    """
    try:
        # Get config path

        #root_path = request.form.get('root_path', osp.abspath(osp.join(__file__, osp.pardir)))
        root_path = "./"
        save_img = request.form.get('save_img', 'true').lower() == 'true'

        # Update save_img option
        if 'val' in OPT:
            OPT['val']['save_img'] = save_img
        
        # Create test dataset and dataloader
        test_loaders = []
        for _, dataset_opt in sorted(OPT['datasets'].items()):
            # If image file is provided, update dataroot_lq
            if 'image' in request.files:
                file = request.files['image']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    temp_dir = tempfile.mkdtemp()
                    filepath = osp.join(temp_dir, filename)
                    file.save(filepath)
                    dataset_opt['dataroot_lq'] = temp_dir

            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=OPT['num_gpu'],
                dist=OPT['dist'], sampler=None, seed=OPT['manual_seed'])
            test_loaders.append(test_loader)
        
        # Run validation
        results = []
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            MODEL.validation(test_loader, current_iter=OPT['name'],
                           tb_logger=None, save_img=save_img)
            results.append({
                'dataset': test_set_name,
                'num_images': len(test_loader.dataset),
                'output_path': OPT['path']['visualization'] if save_img else None
            })
        
        return jsonify({
            'status': 'success',
            'results': results,
        })

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        return jsonify({'error': str(e)}), 500


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



























