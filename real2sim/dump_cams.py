# Runs INSIDE krabby-matcha container, cwd /opt/MAtCha/2d-gaussian-splatting.
# Emulates render_multires.py's scene construction, dumps the cameras the
# GaussianExtractor would integrate with.
import sys, json
sys.path.append("/opt/MAtCha/2d-gaussian-splatting")
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
args = get_combined_args(parser)
dataset = model.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
out = []
for cam in scene.getTrainCameras():
    wvt = cam.world_view_transform.cpu().numpy()   # (4,4)
    out.append({"image_name": cam.image_name,
                "R": cam.R.tolist(), "T": cam.T.tolist(),
                "world_view_transform": wvt.tolist(),
                "camera_center": cam.camera_center.cpu().numpy().tolist(),
                "FoVx": float(cam.FoVx), "FoVy": float(cam.FoVy)})
with open("/work/extract_cameras.json", "w") as f:
    json.dump(out, f)
print("dumped", len(out), "cameras")
