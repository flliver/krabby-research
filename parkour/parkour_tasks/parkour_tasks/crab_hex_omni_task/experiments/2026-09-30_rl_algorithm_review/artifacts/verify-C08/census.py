import yaml, glob, os, collections, json
root='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl'
class L(yaml.SafeLoader): pass
def anyc(loader, suffix, node):
    if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)
L.add_multi_constructor('', anyc)
L.add_multi_constructor('tag:yaml.org,2002:python/', anyc)
rows=[]
for a in sorted(glob.glob(root+'/**/params/agent.yaml', recursive=True)):
    d=os.path.dirname(os.path.dirname(a))
    ag=yaml.load(open(a),Loader=L)
    try:
        ev=yaml.load(open(os.path.join(d,'params/env.yaml')),Loader=L)
        ne=ev['scene']['num_envs']
    except Exception as e: ne=repr(e)[:60]
    al=ag['algorithm']; po=ag['policy']
    rows.append(dict(run=os.path.relpath(d,root),cls=al.get('class_name'),ne=ne,steps=ag.get('num_steps_per_env'),
      ep=al.get('num_learning_epochs'),mb=al.get('num_mini_batches'),kl=al.get('desired_kl'),sch=al.get('schedule'),
      clip=al.get('clip_param'),g=al.get('gamma'),lam=al.get('lam'),gn=al.get('max_grad_norm'),
      en=ag.get('empirical_normalization'),nst=po.get('noise_std_type'),ent=al.get('entropy_coef'),
      lr=al.get('learning_rate'),maxit=ag.get('max_iterations'),pcls=po.get('class_name'),
      sdstd=po.get('state_dependent_std'),stdmax=po.get('action_std_max'),init=po.get('init_noise_std'),
      dist=po.get('distribution_cfg', po.get('distribution')), clipa=ag.get('clip_actions'),vclip=al.get('use_clipped_value_loss'),
      vcoef=al.get('value_loss_coef'), advmb=al.get('normalize_advantage_per_mini_batch'), opt=al.get('optimizer')))
print('total',len(rows))
print(collections.Counter(r['cls'] for r in rows))
ppo=[r for r in rows if r['cls']!='DistillationWithExtractor']
print('ppo',len(ppo))
for k in ['ne','steps','ep','mb','kl','sch','clip','g','lam','gn','en','nst','ent','lr','pcls','sdstd','stdmax','init','dist','clipa','vclip','vcoef','advmb','opt']:
    print(k, collections.Counter(str(r[k]) for r in ppo))
print('ent!=0.01', [(r['run'],r['ent'],r['maxit']) for r in ppo if r['ent']!=0.01])
print('ne!=256', [(r['run'],r['ne']) for r in ppo if r['ne']!=256])
dis=[r for r in rows if r['cls']=='DistillationWithExtractor']
for k in ['ne','steps','ep','mb']:
    print('dist',k, collections.Counter(str(r[k]) for r in dis))
json.dump(rows,open('/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/verify-C08/rows.json','w'),default=str,indent=0)
