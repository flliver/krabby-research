# CPU replay (step-8 precondition)

Read 2026-11-07T17:21:20; dose checked: True.

| item | exact | launch checks clean | detail |
|---|---|---|---|
| (a) sweep table.json + teacher.json | True | True | 91 cell-seeds; closing {'status': 'match', 'matches': ['A', 'B']} |
| (b) Task 4 F19 sel | True | True | pick c28000, ranked ['c28000', 'c29000', 'c30991', 'c27000', 'c30000'], failed blocks [] |
| (c) F19 p3a zero-kg (its 2999) | True | True | FAIL, max margin used 2.7098692720929014 |
| (c) F19 p3b zero-kg (its 5998) | True | True | PASS, max margin used 0.6619605845918585 |
