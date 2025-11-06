import sys
sys.path.append("Restormer/")

from flask import Flask, request, jsonify, send_file
import torch
import torch.nn.functional as F
import os
import yaml
from runpy import run_path
from skimage import img_as_ubyte
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
import traceback


app = Flask(__name__)

TASK = "Single_Image_Defocus_Deblurring"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


SUPPORTED_TASKS = [
    'Motion_Deblurring',
    'Single_Image_Defocus_Deblurring',
    'Deraining',
    'Real_Denoising',
    'Gaussian_Gray_Denoising',
    'Gaussian_Color_Denoising'
]


# load config
def load_model_configs(config_path="../../model_services.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_model_configs()

port = cfg["defocus_deblurring"]["Restormer"]["port"]
host = cfg["defocus_deblurring"]["Restormer"]["host"]
weight_dir = cfg["defocus_deblurring"]["Restormer"]["weight_dir"]

class RestormerService:
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_multiple_of = 8
        self.model = self.load_model(TASK)

        logger.info(f"Use device: {self.device}")
        
    def get_weights_and_parameters(self, task):
        parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False
        }
    
        if task == 'Motion_Deblurring':
            weights = os.path.join(weight_dir, "motion_deblurring.pth")
        elif task == 'Single_Image_Defocus_Deblurring':
            weights = os.path.join(weight_dir, 'single_image_defocus_deblurring.pth')
        elif task == 'Deraining':
            weights = os.path.join(weight_dir, 'deraining.pth')
        elif task == 'Real_Denoising':
            weights = os.path.join(weight_dir, 'real_denoising.pth')
            parameters['LayerNorm_type'] = 'BiasFree'
        elif task == 'Gaussian_Color_Denoising':
            weights = os.path.join(weight_dir, 'gaussian_color_denoising_blind.pth')
            parameters['LayerNorm_type'] = 'BiasFree'
        elif task == 'Gaussian_Gray_Denoising':
            weights = os.path.join(weight_dir, 'gaussian_gray_denoising_blind.pth')
            parameters['inp_channels'] = 1
            parameters['out_channels'] = 1
            parameters['LayerNorm_type'] = 'BiasFree'
        else:
            raise ValueError(f"Not supported task {task}")
        
        return weights, parameters


    def load_model(self, task):
        if task in self.models:
            logger.info(f"model {task} has been loaded, use cache")
            return self.models[task]

        try:
            weights, parameters = self.get_weights_and_parameters(task)
            
            load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
            model = load_arch['Restormer'](**parameters)
            model.to(self.device)
            
            checkpoint = torch.load(weights, map_location=self.device)
            model.load_state_dict(checkpoint['params'])
            model.eval()
            
            self.models[task] = model
            logger.info(f"Succeed loading model: {task}")
            return model
            
        except Exception as e:
            logger.error(f"Failed loading model: {task}")
            raise
    
    def load_img_from_bytes(self, img_bytes, is_gray=False):
        nparr = np.frombuffer(img_bytes, np.uint8)
        if is_gray:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_img_from_bytes(self, img_bytes, is_gray=False):
        nparr = np.frombuffer(img_bytes, np.uint8)
        if is_gray:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def img_to_bytes(self, img, is_gray=False):
        if is_gray:
            success, encoded_img = cv2.imencode('.png', img)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            success, encoded_img = cv2.imencode('.png', img_bgr)
        
        if not success:
            raise ValueError("Img_to_bytes failed")
        
        return encoded_img.tobytes()

    def restore_image(self, img_bytes, task, tile=None, tile_overlap=32):
        
        is_gray = (task == 'Gaussian_Gray_Denoising')
        img = self.load_img_from_bytes(img_bytes, is_gray)
        
        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H = ((height + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((width + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - height if height % self.img_multiple_of != 0 else 0
        padw = W - width if width % self.img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        
        with torch.no_grad():
            if tile is None:
                ## Testing on the original resolution image
                restored = self.model(input_)
            else:
                # test the image tile by tile
                b, c, h, w = input_.shape
                tile = min(tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                
                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h, w).type_as(input_)
                W = torch.zeros_like(E)
                
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = self.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)
                        
                        E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                        W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                restored = E.div_(W)
        
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :height, :width]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        
        result_bytes = self.img_to_bytes(restored, is_gray)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result_bytes


# Global instance
restormer_service = RestormerService()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'restormer-inference',
        'device': str(restormer_service.device),
        'loaded_models': list(restormer_service.models.keys())
    }), 200


@app.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify({
        'tasks': SUPPORTED_TASKS,
        'descriptions': {
            'Motion_Deblurring': '',
            'Single_Image_Defocus_Deblurring': '',
            'Deraining': '',
            'Real_Denoising': '',
            'Gaussian_Gray_Denoising': '',
            'Gaussian_Color_Denoising': ''
        }
    }), 200


@app.route('/restore', methods=['POST'])
def restore():
    """
    -request method 1:
        form-data:
            - image: image file
            - task: task type
            - tile:
            - tile_overlap:
    
    -request method 2:
        JSON:
        {
            "image_base64": "base64 image",
            "task": task type,
            "tile":
            "tile_overlap":
        }
    return:
      - restored image (image/png)
    """
    try:
        task = TASK
        print("task: ", task)

        if task not in SUPPORTED_TASKS:
            return jsonify({
                'error': f'unsupported task: {task}',
                'supported_tasks': SUPPORTED_TASKS
            }), 400

        img_bytes = None

        if request.is_json:
            data = request.get_json()
            image_base64 = data.get('image_base64')
            if not image_base64:
                return jsonify({'error': f'Lack of base64 parameters: {str(e)}'}), 400

            try:
                img_bytes = base64.b64decode(image_base64)
            except Exception as e:
                return jsonify({'error': f'Failed base64 decoding: {str(e)}'}), 400

            tile = data.get('tile')
            tile_overlap = data.get('tile_overlap', 32)
        else:
            if 'image' not in request.files:
                return jsonify({'error': f'Lack of image file: {str(e)}'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': f'No file: {str(e)}'}), 400

            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'unsupported file type',
                    'allowed_types': list(ALLOWED_EXTENSIONS)
                }), 400
            img_bytes = file.read()
            tile = request.form.get('tile')
            tile_overlap = request.form.get('tile_overlap', 32)

        tile = int(tile) if tile else None
        tile_overlap = int(tile_overlap)

        logger.info(f"Execute task: {task}, tile: {tile}, tile_overlap: {tile_overlap}")

        result_bytes = restormer_service.restore_image(img_bytes, task, tile, tile_overlap)

        return send_file(
            io.BytesIO(result_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='restored.png'
        )

    except ValueError as e:
        logger.error(f"Wrong parameter: {str(e)}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Failed image processing: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Failed image restoration',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )


