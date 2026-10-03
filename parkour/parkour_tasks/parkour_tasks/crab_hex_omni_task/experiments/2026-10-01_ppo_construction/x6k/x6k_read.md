# x6k: B to 6000 iterations on C_lr -- the pre-registered reading (CHARTER 10)

Hands L / R: class (feet = mean of the 0.22 and 0.25 holds). model_2999 is the step-3 head.

| fork | role | model_2999 | model_3100 | model_4000 | model_5000 | head (5998) | rise L / R | fall | dose cam/rod (last 100) | priv_reg | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_lr__B_x6k_s11 | one-FAST | FAST 0.240 / INTE 0.149 | FAST 0.245 / INTE 0.144 | FAST 0.247 / FAST 0.222 | FAST 0.250 / FAST 0.244 | FAST 0.237 / FAST 0.224 | -0.003 / +0.075 | 0.000 | 0.27/1.00 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s12 | FAST/FAST | FAST 0.259 / FAST 0.193 | FAST 0.250 / FAST 0.224 | FAST 0.238 / FAST 0.210 | FAST 0.258 / FAST 0.250 | FAST 0.260 / FAST 0.262 | +0.002 / +0.069 | 0.000 | 0.28/1.00 | 0.1000 | STABLE; **candidate** |
| C_lr__B_x6k_s13 | one-FAST | FAST 0.228 / INTE 0.137 | FAST 0.233 / INTE 0.148 | FAST 0.232 / FAST 0.196 | FAST 0.231 / FAST 0.236 | FAST 0.240 / FAST 0.239 | +0.012 / +0.101 | 0.003 | 0.30/1.00 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s14 | one-FAST | FAST 0.231 / INTE 0.144 | FAST 0.249 / INTE 0.161 | FAST 0.248 / FAST 0.250 | FAST 0.261 / FAST 0.255 | FAST 0.266 / FAST 0.251 | +0.035 / +0.107 | 0.000 | 0.28/1.03 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s15 | one-FAST | INTE 0.165 / FAST 0.248 | INTE 0.163 / FAST 0.242 | FAST 0.174 / FAST 0.262 | FAST 0.188 / FAST 0.235 | FAST 0.219 / FAST 0.255 | +0.055 / +0.007 | 0.000 | 0.31/1.07 | 0.1000 | completes; **candidate** |
| C_lr__C0_x6k_s11 | C0 control | SLOW 0.084 / SLOW 0.081 | SLOW 0.085 / SLOW 0.080 | SLOW 0.086 / SLOW 0.082 | SLOW 0.086 / SLOW 0.080 | SLOW 0.086 / SLOW 0.082 | +0.001 / +0.001 | 0.025 | 6.71/9.77 | 0.1000 | FROZEN |

**Reading: LATE FORMATION** -- 4/4 one-FAST forks complete: ['C_lr__B_x6k_s11', 'C_lr__B_x6k_s13', 'C_lr__B_x6k_s14', 'C_lr__B_x6k_s15']

- s12 (FAST/FAST head): STABLE
- C0 control: FROZEN
- two-direction candidates (FAST/FAST, not tripped, dose ok): ['C_lr__B_x6k_s11', 'C_lr__B_x6k_s12', 'C_lr__B_x6k_s13', 'C_lr__B_x6k_s14', 'C_lr__B_x6k_s15']
