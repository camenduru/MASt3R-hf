# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# masst3r demo
# --------------------------------------------------------
import spaces
import os
import sys
import os.path as path
import torch
import tempfile

HERE_PATH = path.normpath(path.dirname(__file__))  # noqa
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, './mast3r'))  # noqa
sys.path.insert(0, MASt3R_REPO_PATH)  # noqa

import mast3r.demo
mast3r.demo.get_reconstructed_scene = spaces.GPU(mast3r.demo.get_reconstructed_scene)
mast3r.demo.get_3D_model_from_scene = spaces.GPU(mast3r.demo.get_3D_model_from_scene)

from mast3r.demo import main_demo
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import matplotlib.pyplot as pl
pl.ion()

# for gpu >= Ampere and pytorch >= 1.12
torch.backends.cuda.matmul.allow_tf32 = True
batch_size = 1

weights_path = "naver/" + 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
device = 'cuda'
model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
chkpt_tag = hash_md5(weights_path)

# mast3r will write the 3D model inside tmpdirname/chkpt_tag
with tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') as tmpdirname:
    cache_path = os.path.join(tmpdirname, chkpt_tag)
    os.makedirs(cache_path, exist_ok=True)
    main_demo(tmpdirname, model, device, 512, server_name=None, server_port=None,
              silent=True, share=None, gradio_delete_cache=7200)
