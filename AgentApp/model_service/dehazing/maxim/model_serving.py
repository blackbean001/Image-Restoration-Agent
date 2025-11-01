
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
