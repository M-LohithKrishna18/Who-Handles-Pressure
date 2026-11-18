"""Stage 6: two-stage selection.
 A. Candidate pool (rules):  sample-eligible (60 pressure balls, 8 pressure innings); available (not retained,
    not PAK/BAN, not inactive/retired, not banned - news-checked); in form (2025-26 RAA/100 >= -25);
    evidence mostly from IPL or full-member T20Is (>= 50% of pressure balls); bowlers: >= 24 IPL pressure balls.
 B. Ranking within pool: rating (pressure RAA/100, shrunk toward own record), with bootstrap stability."""
import os
import pandas as pd, numpy as np
from s4_ratings import build, OUT
ASSOC = {272, 283, 311, 312, 313, 282, 24, 26, 89, 315, 3886}
FORM_MIN, CORE_SHARE, IPL_MIN_BOWL = -25, 0.5, 24
bat, bowl = build()
for role, df in (('bat', bat), ('bowl', bowl)):
    t = pd.read_csv(OUT + f'{role}_ratings_all.csv', index_col=0)
    p = df[df.pressure]
    opp = p.bowling_team_id if role == 'bat' else p.batting_team_id
    core = (p.comp_group == 'IPL') | ((p.comp_group == 'T20I') & ~opp.isin(ASSOC))
    t['core_share'] = core.groupby(p.pid).mean().reindex(t.index)
    t['ipl_p_balls'] = (p.comp_group == 'IPL').groupby(p.pid).sum().reindex(t.index).fillna(0).astype(int)
    t['t20i_p_balls'] = (p.comp_group == 'T20I').groupby(p.pid).sum().reindex(t.index).fillna(0).astype(int)
    t['rule_available'] = ~t.excluded.fillna(False).astype(bool)
    t['rule_form'] = t.form_2526_raa100.fillna(0) >= FORM_MIN
    t['rule_core'] = t.core_share >= CORE_SHARE
    t['rule_ipl'] = (t.ipl_p_balls >= IPL_MIN_BOWL) if role == 'bowl' else True
    t['in_pool'] = t.eligible_sample & t.rule_available & t.rule_form & t.rule_core & t.rule_ipl
    t.to_csv(OUT + f'{role}_ratings_all.csv')
    pool = t[t.in_pool].sort_values('rating', ascending=False)
    pool.to_csv(OUT + f'{role}_pool.csv')
    print(f'\n===== {role.upper()} pool: {len(pool)} players')
    print(pool.head(16)[['name', 'rating', 'ci_lo', 'ci_hi', 'p_balls', 'ipl_p_balls', 'core_share', 'form_2526_raa100']].round(2).to_string())
