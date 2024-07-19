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
import gradio
import shutil

HERE_PATH = path.normpath(path.dirname(__file__))  # noqa
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, './mast3r'))  # noqa
sys.path.insert(0, MASt3R_REPO_PATH)  # noqa

from mast3r.demo import get_reconstructed_scene, get_3D_model_from_scene, set_scenegraph_options
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import matplotlib.pyplot as pl
pl.ion()

# for gpu >= Ampere and pytorch >= 1.12
torch.backends.cuda.matmul.allow_tf32 = True
batch_size = 1

weights_path = "naver/" + 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
chkpt_tag = hash_md5(weights_path)

tmpdirname = tempfile.mkdtemp(suffix='_mast3r_gradio_demo')
image_size = 512
silent = True
gradio_delete_cache = 7200


@spaces.GPU()
def local_get_reconstructed_scene(current_scene_state,
                                  filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                                  win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    return get_reconstructed_scene(tmpdirname, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                                   filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                   as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                                   win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw)


@spaces.GPU()
def local_get_3D_model_from_scene(scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                                  clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    return get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                   clean_depth, transparent_cams, cam_size, TSDF_thresh)


recon_fun = local_get_reconstructed_scene
model_from_scene_fun = local_get_3D_model_from_scene


def get_context(delete_cache):
    css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
    title = "MASt3R Demo"
    if delete_cache:
        return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
    else:
        return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions


with get_context(gradio_delete_cache) as demo:
    # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
    scene = gradio.State(None)
    gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
    with gradio.Column():
        inputfiles = gradio.File(file_count="multiple")
        with gradio.Row():
            with gradio.Column():
                with gradio.Row():
                    lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                    niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
                                           label="num_iterations", info="For coarse alignment!")
                    lr2 = gradio.Slider(label="Fine LR", value=0.014, minimum=0.005, maximum=0.05, step=0.001)
                    niter2 = gradio.Number(value=200, precision=0, minimum=0, maximum=100_000,
                                           label="num_iterations", info="For refinement!")
                    optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                  value='refine', label="OptLevel",
                                                  info="Optimization level")
                with gradio.Row():
                    matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=5.,
                                                      minimum=0., maximum=30., step=0.1,
                                                      info="Before Fallback to Regr3D!")
                    shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                        info="Only optimize one set of intrinsics for all views")
                    scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                       ("swin: sliding window", "swin"),
                                                       ("logwin: sliding window with long range", "logwin"),
                                                       ("oneref: match one image with all", "oneref")],
                                                      value='complete', label="Scenegraph",
                                                      info="Define how to make pairs",
                                                      interactive=True)
                    with gradio.Column(visible=False) as win_col:
                        winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                minimum=1, maximum=1, step=1)
                        win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                    refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                          minimum=0, maximum=0, step=1, visible=False)
        run_btn = gradio.Button("Run")

        with gradio.Row():
            # adjust the confidence threshold
            min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
            # adjust the camera size in the output pointcloud
            cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
            TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
        with gradio.Row():
            as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
            # two post process implemented
            mask_sky = gradio.Checkbox(value=False, label="Mask sky")
            clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
            transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

        outmodel = gradio.Model3D()

        # events
        scenegraph_type.change(set_scenegraph_options,
                               inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                               outputs=[win_col, winsize, win_cyclic, refid])
        inputfiles.change(set_scenegraph_options,
                          inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                          outputs=[win_col, winsize, win_cyclic, refid])
        win_cyclic.change(set_scenegraph_options,
                          inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                          outputs=[win_col, winsize, win_cyclic, refid])
        run_btn.click(fn=recon_fun,
                      inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                              as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                              scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                      outputs=[scene, outmodel])
        min_conf_thr.release(fn=model_from_scene_fun,
                             inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                     clean_depth, transparent_cams, cam_size, TSDF_thresh],
                             outputs=outmodel)
        cam_size.change(fn=model_from_scene_fun,
                        inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                clean_depth, transparent_cams, cam_size, TSDF_thresh],
                        outputs=outmodel)
        TSDF_thresh.change(fn=model_from_scene_fun,
                           inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                   clean_depth, transparent_cams, cam_size, TSDF_thresh],
                           outputs=outmodel)
        as_pointcloud.change(fn=model_from_scene_fun,
                             inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                     clean_depth, transparent_cams, cam_size, TSDF_thresh],
                             outputs=outmodel)
        mask_sky.change(fn=model_from_scene_fun,
                        inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                clean_depth, transparent_cams, cam_size, TSDF_thresh],
                        outputs=outmodel)
        clean_depth.change(fn=model_from_scene_fun,
                           inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                   clean_depth, transparent_cams, cam_size, TSDF_thresh],
                           outputs=outmodel)
        transparent_cams.change(model_from_scene_fun,
                                inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                        clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                outputs=outmodel)
demo.launch(share=None, server_name=None, server_port=None)
shutil.rmtree(tmpdirname)
