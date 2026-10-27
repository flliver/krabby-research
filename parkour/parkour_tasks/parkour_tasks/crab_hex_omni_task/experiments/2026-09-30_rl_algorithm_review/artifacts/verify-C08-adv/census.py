import glob, os, yaml, collections, re, json, csv
root='/home/nickmagus/krabby/krabby-research/parkour/logs'
rows=[]
for a in sorted(glob.glob(root+'/**/params/agent.yaml', recursive=True)):
    d=os.path.dirname(os.path.dirname(a))
    ag=yaml.load(open(a), Loader=yaml.UnsafeLoader)
    txt=open(os.path.join(os.path.dirname(a),'env.yaml')).read()
    m=re.search(r'^scene:\n(?:  .*\n)*?  num_envs: (\d+)', txt, re.M)
    al=ag.get('algorithm',{}) or {}
    rows.append(dict(run=d.replace(root+'/rsl_rl/',''), algo=al.get('class_name'), num_envs=int(m.group(1)), maxit=ag.get('max_iterations'), ent=al.get('entropy_coef'), ntb=len(glob.glob(d+'/events.out.tfevents*'))))
json.dump(rows, open('/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/verify-C08-adv/rows.json','w'))
r=list(csv.DictReader(open('/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/telemetry/runs.csv')))
s={x['experiment']+'/'+x['run'] for x in r}
for x in rows:
    if x['run'] not in s: print('missing in telemetry', x)
for x in r:
    if x['alg_class']=='PPOWithExtractor' and x['num_envs']=='64': print(x['experiment'],x['run'],'iters_logged',x['iters_logged'],'maxit',x['max_iterations'])
