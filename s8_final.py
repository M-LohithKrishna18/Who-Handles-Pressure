"""Stage 8: final score = 0.8 x pressure rating + 0.2 x recent form (2025-26 RAA/100, shrunk with 300-ball prior).
Both in runs above average per 100 balls -> same units (no BASRA-style mixing). Top 10 per list from the stage-6 pool."""
import os
import pandas as pd
OUT = os.environ.get('OUT_DIR', 'out/')
W_P, W_F, FORM_K = 0.8, 0.2, 300
for role in ('bat', 'bowl'):
    t = pd.read_csv(OUT + f'{role}_pool.csv', index_col=0)
    bs = pd.read_csv(OUT + f'{role}_bootstrap.csv', index_col=0)[['top10_pct']]
    n = t.form_2526_balls.fillna(0)
    t['form_shrunk'] = t.form_2526_raa100.fillna(0) * n / (n + FORM_K)
    t['final_score'] = W_P * t.rating + W_F * t.form_shrunk
    t = t.join(bs).sort_values('final_score', ascending=False)
    t.insert(0, 'rank', range(1, len(t) + 1))
    t.to_csv(OUT + f'final_{role}_list.csv')
    print(f'\n===== FINAL {role.upper()}')
    print(t.head(12)[['rank', 'name', 'final_score', 'rating', 'form_2526_raa100', 'p_balls', 'ipl_p_balls', 'top10_pct']].round(1).to_string(index=False))
