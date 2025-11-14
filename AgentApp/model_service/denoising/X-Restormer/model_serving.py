import sys
sys.path.append("X-Restormer")

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import os.path as osp
import torch
import yaml
import logging
import tempfile
import shutil
from io import BytesIO
from PIL import Image
import base64

import xrestormer.archs
import xrestormer.data
import xrestormer.models
from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, make_exp_dirs
from basicsr.utils.options import ordered_yaml, _postprocess_yml_value
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.sr_model import SRModel


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()
port = cfg["denoising"]["XRestormer"]["port"]
host = cfg["denoising"]["XRestormer"]["host"]
opt_config = cfg["denoising"]["XRestormer"]["config"]
model_path = cfg["denoising"]["XRestormer"]["model_path"]


# global configuration
model = None
opt = None
logger = None

def load_config(config_path):
    with open(config_path, mode='r') as f:
        config = yaml.load(f, Loader=ordered_yaml()[0])
    return config


def initialize_model(opt_config):
    global model, opt, logger

    opt = load_config(opt_config)
    print("opt: ", opt)

    opt['path']['pretrain_network_g'] = model_path

    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt['is_train'] = False

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']

    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    
    results_root = opt['path'].get('results', './results')
    opt['path']['results_root'] = osp.join(results_root, opt["name"])
    opt['path']['log'] = osp.join(results_root, opt["name"])
    opt['path']['visualization'] = osp.join(results_root, opt["name"])

    make_exp_dirs(opt)

    log_file = osp.join(opt['path']['log'], f"service_{opt['name']}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info("Model initialization started")

    torch.backends.cudnn.benchmark = True
    model = build_model(opt)

    logger.info("Model loaded successfully")
    return model, opt


def process_image(image_path, output_path):
    global model, opt, logger
    
    if model is None:
        raise ValueError("Model not initialized")
    
    try:
        temp_dataset_opt = opt['datasets']['test'].copy()
        temp_dataset_opt['dataroot_lq'] = osp.dirname(image_path)

        temp_dataset_opt['io_backend'] = {'type': 'disk'}
    except Exception as e:
        logger.error(f"Error creating data: {str(e)}")

    try:
        from basicsr.data.paired_image_dataset import PairedImageDataset
        from torch.utils.data import DataLoader
        print("temp_dataset_opt: ", temp_dataset_opt)

        test_set = build_dataset(temp_dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        model.validation(test_loader, current_iter=opt['name'], 
                        tb_logger=None, save_img=True)
        
        logger.info(f"Image processed successfully: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/process', methods=['POST'])
def process():
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    fileStorage = request.files['image']

    if fileStorage.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        filename = secure_filename(fileStorage.filename)
        
        output_dir = "results"

        os.makedirs(output_dir, exist_ok=True)
        output_path = osp.join(output_dir, filename)

        success = process_image(osp.join('inputs', filename), output_path)
        
        # the path is determined by basicsr
        output_image_path = f"{output_dir}/{opt['name']}/test/{filename.split('.')[0]}.png"
        if success and osp.exists(output_image_path):
            with open(output_image_path, 'rb') as f:
                img_data = f.read()
            
            # the image file appears both at output_image_path and f'restored_{filename}'
            return send_file(
                BytesIO(img_data),
                mimetype='image/png',
                as_attachment=True,
                download_name=f'restored_{filename}'
            )

        else:
            return jsonify({'error': 'Image processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Error in process endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    try:
        initialize_model(opt_config)
        print("Model initialized successfully!")
        print(f"Starting Flask server on {host}:{port}")

        app.run(host=host, port=port, debug=False)

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise


