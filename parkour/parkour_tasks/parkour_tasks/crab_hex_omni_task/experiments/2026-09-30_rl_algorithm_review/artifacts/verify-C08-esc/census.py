import yaml, glob, os, collections, json, re, sys
root='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl'
class L(yaml.SafeLoader): pass
def anyc(loader, suffix, node):
    if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)
L.add_multi_constructor('', anyc)
L.add_multi_constructor('tag:yaml.org,2002:python/', anyc)
rows=[]
# also count run dirs that have tfevents but no agent.yaml
tf_dirs=set(os.path.dirname(p) for p in glob.glob(root+'/**/events.out.tfevents*', recursive=True))
ag_dirs=set()
for a in sorted(glob.glob(root+'/**/params/agent.yaml', recursive=True)):
    d=os.path.dirname(os.path.dirname(a)); ag_dirs.add(d)
    ag=yaml.load(open(a),Loader=L)
    ev=yaml.load(open(os.path.join(d,'params/env.yaml')),Loader=L)
    ne=ev['scene']['num_envs']
    al=ag['algorithm']; po=ag['policy']
    sym=al.get('symmetry_cfg')
    rows.append(dict(run=os.path.relpath(d,root),cls=al.get('class_name'),ne=ne,steps=ag.get('num_steps_per_env'),
      ep=al.get('num_learning_epochs'),mb=al.get('num_mini_batches'),kl=al.get('desired_kl'),sch=al.get('schedule'),
      clip=al.get('clip_param'),g=al.get('gamma'),lam=al.get('lam'),gn=al.get('max_grad_norm'),
      en=ag.get('empirical_normalization'),nst=po.get('noise_std_type'),ent=al.get('entropy_coef'),
      lr=al.get('learning_rate'),maxit=ag.get('max_iterations'),pcls=po.get('class_name'),
      sdstd=po.get('state_dependent_std'),stdmax=po.get('action_std_max'),stdreset=po.get('action_std_reset'),init=po.get('init_noise_std'),
      clipa=ag.get('clip_actions'),vclip=al.get('use_clipped_value_loss'),vcoef=al.get('value_loss_coef'),
      advmb=al.get('normalize_advantage_per_mini_batch'),opt=al.get('optimizer'),
      sym_aug=(sym or {}).get('use_data_augmentation') if isinstance(sym,dict) else None,
      sym_mirror=(sym or {}).get('use_mirror_loss') if isinstance(sym,dict) else None,
      ntf=len(glob.glob(d+'/events.out.tfevents*')), resume=ag.get('resume'), seed=ag.get('seed')))
print('tfevent dirs',len(tf_dirs),'agent.yaml dirs',len(ag_dirs),'tf without yaml',sorted(os.path.relpath(x,root) for x in tf_dirs-ag_dirs))
print('total',len(rows)); print(collections.Counter(r['cls'] for r in rows))
ppo=[r for r in rows if r['cls']=='PPOWithExtractor']
print('ppo',len(ppo))
for k in ['ne','steps','ep','mb','kl','sch','clip','g','lam','gn','en','nst','ent','lr','pcls','sdstd','stdmax','stdreset','init','clipa','vclip','vcoef','advmb','opt','sym_aug','sym_mirror','resume']:
    print(k, dict(collections.Counter(str(r[k]) for r in ppo)))
print('ent!=0.01', [(r['run'],r['ent'],r['maxit'],r['ne']) for r in ppo if r['ent']!=0.01])
print('ne!=256', sorted([(r['run'],r['ne'],r['maxit']) for r in ppo if r['ne']!=256]))
print('stdmax set', [(r['run'],r['stdmax'],r['stdreset'],r['maxit'],r['ne']) for r in ppo if r['stdmax'] is not None])
print('clipa!=1', [(r['run'],r['clipa'],r['maxit']) for r in ppo if r['clipa']!=1.0])
print('init!=1.5', collections.Counter((r['init']) for r in ppo if r['init']!=1.5), [r['run'] for r in ppo if r['init']!=1.5][:5])
print('lr', collections.Counter(r['lr'] for r in ppo))
dis=[r for r in rows if r['cls']=='DistillationWithExtractor']
print('distill ne', collections.Counter(r['ne'] for r in dis))
json.dump(rows,open(os.path.dirname(os.path.abspath(__file__))+'/rows.json','w'),default=str,indent=0)
