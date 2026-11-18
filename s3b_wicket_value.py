"""Stage 3b: wicket value in chases from win probability.
Runs-remaining value (DLS table) undervalues death wickets in a close chase. In the 2nd innings we convert
the win-probability cost of a wicket into runs:  value = [WP(dot) - WP(wicket)] / [WP(single) - WP(dot)]
(i.e. how many runs the wicket is 'worth' at that moment). Used when the denominator is meaningful
(> 0.2 win-% points per run); clipped to [runs-remaining value, 30]. First innings keeps the runs-remaining value."""
import os
import pandas as pd, numpy as np, pickle
from wp_features import feats
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_expected.pkl')
models = pickle.load(open(OUT + 'wp_final_models.pkl', 'rb'))

def wp2(d):
    out = np.full(len(d), np.nan)
    won = (d.runs_needed <= 0).values; lost = ~won & ((d.wkts_in_hand <= 0) | (d.balls_left <= 0)).values
    out[won], out[lost] = 1.0, 0.0
    live = np.isnan(out)
    x = d[live].copy(); x['balls_left'] = x.balls_left.clip(lower=1)
    out[live] = models[2].predict_proba(feats(x, 2))[:, 1]
    return out

c = b[b.innings == 2]
def step(r, w):
    h = c.copy(); h['runs_needed'] -= r; h['runs_before'] += r; h['legal_before'] += 1
    h['balls_left'] -= 1; h['wkts_in_hand'] -= w; return wp2(h)
dot, one, wk = step(0, 0), step(1, 0), step(0, 1)
den = one - dot
val = np.where(den > 0.002, (dot - wk) / np.maximum(den, 1e-6), np.nan)
b['wkt_value_runs'] = b.wkt_value
b.loc[c.index, 'wkt_value'] = np.where(np.isnan(val), c.wkt_value, np.clip(val, c.wkt_value, 30))
b.to_pickle(OUT + 'balls_expected.pkl')
print('Wicket value (runs) by innings x phase, before -> after:')
print(b.groupby(['innings', 'phase']).agg(before=('wkt_value_runs', 'mean'), after=('wkt_value', 'mean')).round(1))
