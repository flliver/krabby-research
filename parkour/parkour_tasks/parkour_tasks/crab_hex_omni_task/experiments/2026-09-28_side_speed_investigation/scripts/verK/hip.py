import sys,json,numpy as np,math
sys.path.insert(0,"/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp")
import crab_hex_linkage as L, torch
print("hip default deg",math.degrees(L.hip_default_rad()),"knee def L",math.degrees(L.knee_default_left_rad()))
for s in [-1,1]:
  th=torch.tensor(L.hip_default_rad()+0.25*s,dtype=torch.float64)
  lam=L.hip_actuator_length(th); print("hip clip",s,"len",float(lam),"min/max",L.HIP_LEN_MIN_M,L.HIP_LEN_MAX_M)
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import STAMPS, jidx
for tag,lab in [("W","lat_l"),("SH",None),("W","fwd")]:
  d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
  ih=jidx(meta,"Hip_Femur_RevoluteJoint")
  hs=[];rs=[]
  if meta.get("suite"): files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",[labels.index(lab)]) for e in meta["env_index_map"] if e["label"]==lab]
  else: files=[(p,[1,2]) for p in sorted((d/"raw").glob("episode_*.npz"))]
  for p,ids in files[:40]:
    z=np.load(p); m=np.isin(z["segment_id"],ids)&z["steady_mask"].astype(bool)
    hs.append(np.degrees(z["joint_pos"][m][:,ih])); rs.append(np.abs(z["rod_speed"][m][:,:6])/L.HIP_SPEED_M_S if "rod_speed" in z.files else np.zeros((1,6)))
  h=np.concatenate(hs); r=np.concatenate(rs)
  print(tag,lab,"hip deg p1/p5/p95/p99",np.percentile(h,[1,5,95,99]).round(1),"hip rod |v|/cap mean",r.mean().round(2),">=0.9",(r>=0.9).mean().round(3))
