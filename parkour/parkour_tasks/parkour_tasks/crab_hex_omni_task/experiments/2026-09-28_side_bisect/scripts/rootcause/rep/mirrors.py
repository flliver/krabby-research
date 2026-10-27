"""Build the omni M_y / M_x maps on CPU from the recorded name lists (read-only; same code the training uses)."""
import importlib.util, sys
import torch
P = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_mirror.py"
spec = importlib.util.spec_from_file_location("omni_mirror_probe", P); om = importlib.util.module_from_spec(spec)
sys.modules["omni_mirror_probe"] = om; spec.loader.exec_module(om)
def build(meta):
    lr_s, fa_s, _ = om.head_signs("raw")
    kw = dict(joint_names=meta["joint_names"], action_joint_names=meta["action_joint_names"],
              contact_body_names=[f"{l}_Footpad" for l in ["FL", "FR", "ML", "MR", "RL", "RR"]],
              scan_nx=9, scan_ny=12, scan_ordering="xy", history_length=10, device="cpu")
    lr = om._fwd.CrabHexMirror(**kw, head_signs=lr_s, joint_sign_rules=om.JOINT_SIGN_RULES_LR, leg_swap=om.LEG_SWAP_LR,
                               scan_axis=om.SCAN_AXIS_LR, vec3_signs=om.VEC3_SIGNS_LR)
    fa = om._fwd.CrabHexMirror(**kw, head_signs=fa_s, joint_sign_rules=om.JOINT_SIGN_RULES_FA, leg_swap=om.LEG_SWAP_FA,
                               scan_axis=om.SCAN_AXIS_FA, vec3_signs=om.VEC3_SIGNS_FA)
    assert lr.obs_dim == 1149, lr.obs_dim
    return lr, fa
