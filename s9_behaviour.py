"""Behaviour lens: off-field / external pressure as it shows on the field (descriptive, not in the rating).
 - Technique drift: Jensen-Shannon distance between a player's shot mix (batters: stroke) or delivery mix
   (bowlers: delivery_style) on pressure balls vs ordinary balls. 0 = identical habits.
 - Control: wide/no-ball rate under pressure vs ordinary (bowlers).
 - Repeat-the-mistake (bowlers): after being hit for a boundary, share of next balls that repeat the same delivery style,
   and runs per ball when they repeat.
 - Repeat-the-mistake (batters): share of dismissals to their single most common delivery style."""
import os
import pandas as pd, numpy as np, json
from scipy.spatial.distance import jensenshannon
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_expected.pkl')
fb = pd.read_csv(OUT + 'final_bat_list.csv', index_col=0).head(10); fw = pd.read_csv(OUT + 'final_bowl_list.csv', index_col=0).head(10)
def jsd(d, col, pid_col, pid):
    x = d[d[pid_col] == pid].dropna(subset=[col])
    if x.pressure.sum() < 30: return np.nan
    p = x[x.pressure][col].value_counts(normalize=True); o = x[~x.pressure][col].value_counts(normalize=True)
    idx = p.index.union(o.index); return float(jensenshannon(p.reindex(idx, fill_value=0), o.reindex(idx, fill_value=0), base=2))
bat = b[b.faced]; R = {}
# league reference: all batters / bowlers with >= 60 pressure balls
def league(d, col, pid_col):
    ids = d[d.pressure].groupby(pid_col).size(); ids = ids[ids >= 60].index
    v = [jsd(d, col, pid_col, i) for i in ids]; return float(np.nanmedian(v)), len(ids)
R['league_bat_shot_drift_median'], R['n_bat'] = league(bat, 'stroke', 'striker_id')
R['league_bowl_delivery_drift_median'], R['n_bowl'] = league(b, 'delivery_style', 'bowler_id')
# league-wide shot mix shift under pressure
sm = bat.dropna(subset=['stroke']).groupby('pressure').stroke.value_counts(normalize=True).unstack(0).fillna(0)
sm['diff_pts'] = (sm[True] - sm[False]) * 100
R['league_shots_more_under_pressure'] = sm.diff_pts.sort_values(ascending=False).head(5).round(1).to_dict()
R['league_shots_less_under_pressure'] = sm.diff_pts.sort_values().head(5).round(1).to_dict()
wide = b.extras_type.isin(['wide', 'Overstep', 'other'])
R['league_wide_nb_rate'] = {'pressure': round(wide[b.pressure].mean() * 100, 2), 'ordinary': round(wide[~b.pressure].mean() * 100, 2)}
# bowler repeat after boundary
s = b.sort_values(['match_id', 'innings', 'innings_ball_num'])
g = s.groupby(['match_id', 'innings'])
s['prev_bdry_same_bowler'] = g.is_bdry.shift(1, fill_value=False) & (g.bowler_id.shift(1) == s.bowler_id) if 'is_bdry' in s else False
s['prev_style'] = g.delivery_style.shift(1)
after = s[s.prev_bdry_same_bowler & s.delivery_style.notna()]
after = after.assign(repeat=after.delivery_style == after.prev_style)
R['league_after_boundary'] = {'repeat_share_pct': round(after.repeat.mean() * 100, 1),
    'runs_per_ball_if_repeat': round(after[after.repeat].bowl_runs.mean(), 2), 'runs_per_ball_if_change': round(after[~after.repeat].bowl_runs.mean(), 2)}
rows_b = []
for pid, r in fb.iterrows():
    x = bat[bat.striker_id == pid]; outs = x[x.striker_out].delivery_style.value_counts()
    rows_b.append(dict(name=r['name'], shot_drift=round(jsd(bat, 'stroke', 'striker_id', pid), 3),
                       top_dismissal_style=outs.index[0] if len(outs) else '', top_style_share_pct=round(outs.iloc[0] / outs.sum() * 100, 0) if len(outs) else np.nan, dismissals=int(outs.sum())))
rows_w = []
for pid, r in fw.iterrows():
    x = b[b.bowler_id == pid]; w = x.extras_type.isin(['wide', 'Overstep', 'other'])
    a = after[after.bowler_id == pid]
    rows_w.append(dict(name=r['name'], delivery_drift=round(jsd(b, 'delivery_style', 'bowler_id', pid), 3),
                       wide_nb_pressure_pct=round(w[x.pressure].mean() * 100, 1), wide_nb_ordinary_pct=round(w[~x.pressure].mean() * 100, 1),
                       repeat_after_boundary_pct=round(a.repeat.mean() * 100, 0) if len(a) >= 10 else np.nan, n_after_boundary=len(a)))
pd.DataFrame(rows_b).to_csv(OUT + 'behaviour_bat.csv', index=False); pd.DataFrame(rows_w).to_csv(OUT + 'behaviour_bowl.csv', index=False)
json.dump(R, open(OUT + 'behaviour_league.json', 'w'), indent=1)
print(json.dumps(R, indent=1)); print(pd.DataFrame(rows_b).to_string(index=False)); print(pd.DataFrame(rows_w).to_string(index=False))
