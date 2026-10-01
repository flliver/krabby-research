import csv, collections, statistics, json, os, glob
SP='/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review'
r=list(csv.DictReader(open(SP+'/telemetry/runs.csv')))
ppo=[x for x in r if x['alg_class']=='PPOWithExtractor']
print('runs.csv rows',len(r),'ppo',len(ppo), collections.Counter(x['num_envs'] for x in ppo))
print('64-env iters_logged', collections.Counter(x['iters_logged'] for x in ppo if x['num_envs']=='64'))
for ne in ['256','64']:
    v=[float(x['fps_median'])*float(x['s_per_iter_median']) for x in ppo if x['num_envs']==ne and x['fps_median'] and x['s_per_iter_median']]
    print(ne,'samples/iter median',round(statistics.median(v),1),'min',round(min(v),1),'max',round(max(v),1),'n',len(v))
for x in ppo:
    if x['action_std_max'] not in ('', None): print('std_max cfg', x['experiment'],x['run'],x['action_std_max'],x['action_std_reset'],'iters',x['iters_logged'],'std_first',x['std_first'],'std_last',x['std_last'],'std_max_tb',x['std_max'])
print('ent!=0.01',[ (x['run'],x['entropy_coef'],x['iters_logged']) for x in ppo if x['entropy_coef']!='0.01'])
print('clip_actions', collections.Counter(x['clip_actions'] for x in ppo))
# real runs longer than 20 iters at 256
long=[x for x in ppo if int(x['iters_logged'])>20]
print('ppo with >20 iters logged', len(long), collections.Counter(x['num_envs'] for x in long))
# rows_all census by voter1
ra=json.load(open(SP+'/verify-C08/rows_all.json'))
print('rows_all total',len(ra), collections.Counter(x.get('cls') for x in ra))
pa=[x for x in ra if x.get('cls')=='PPOWithExtractor']
print('rows_all ppo ne', collections.Counter(str(x.get('ne')) for x in pa))
# run dirs vs runs.csv
root=P='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl'
s={x['experiment']+'/'+x['run'] for x in r}
dirs=[os.path.relpath(os.path.dirname(os.path.dirname(a)),root) for a in glob.glob(root+'/**/params/agent.yaml',recursive=True)]
print('dirs not in runs.csv', [d for d in dirs if d not in s])
print('runs.csv not in dirs', [d for d in s if d not in dirs])
