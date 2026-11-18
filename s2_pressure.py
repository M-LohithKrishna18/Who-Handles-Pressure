"""Stage 2: pressure definition.
Leverage = expected absolute swing in win probability over the possible outcomes of the ball
(outcome mix taken from league frequencies for that innings x phase). Normalised so average ball = 1.
Pressure score = leverage x knockout weight; cut = top X% WITHIN each phase (PP / middle / death).
Phase weights (PP, death x1.25) are carried as rating weights. Super-over balls = automatic pressure.
Build-up situations (batter & bowler) are tagged per ball and tested against ordinary balls."""
import os
import pandas as pd, numpy as np, pickle, json
from wp_features import feats
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_state.pkl')
models = pickle.load(open(OUT + 'wp_final_models.pkl', 'rb'))
TOP = 0.15                       # share of balls in each phase called pressure (tested 10/15/20%)
PHASE_W = {'Powerplay': 1.25, 'Middle': 1.0, 'Death': 1.25}
KO_W = 1.5

over = (b.legal_before // 6 + 1).clip(upper=20)
b['phase'] = np.select([over <= 6, over >= 16], ['Powerplay', 'Death'], 'Middle')

# ---------- outcome mix per innings x phase ----------
b['oc'] = np.select([b.is_wicket, ~b.legal, b.runs >= 6, b.runs >= 4, b.runs == 3, b.runs == 2, b.runs == 1],
                    ['W', 'X', '6', '4', '3', '2', '1'], '0')
mix = b.groupby(['innings', 'phase']).oc.value_counts(normalize=True).rename('p').reset_index()
OUTCOMES = {'0': (0, 1, 0), '1': (1, 1, 0), '2': (2, 1, 0), '3': (3, 1, 0), '4': (4, 1, 0), '6': (6, 1, 0),
            'W': (0, 1, 1), 'X': (1, 0, 0)}          # (runs, legal ball?, wicket)

def wp_state(d, i):
    """WP for batting team in innings i for arbitrary (possibly terminal) states."""
    out = np.full(len(d), np.nan)
    if i == 2:
        won = d.runs_needed <= 0; lost = ~won & ((d.wkts_in_hand <= 0) | (d.balls_left <= 0))
        out[won.values], out[lost.values] = 1.0, 0.0
    else:
        end = (d.wkts_in_hand <= 0) | (d.balls_left <= 0)
        if end.any():   # first innings over -> WP of the *chasing* side at start of chase, flipped
            e = d[end].copy()
            tgt = e.runs_before + 1
            s = pd.DataFrame({'runs_needed': tgt, 'balls_left': e.max_balls2, 'wkts_in_hand': 10, 'max_balls': e.max_balls2,
                              'par': e.par, 'elo_diff': -e.elo_diff, 'is_ipl': e.is_ipl, 'is_ateam': e.is_ateam,
                              'impact_era': e.impact_era, 'runs_before': 0, 'legal_before': 0}, index=e.index)
            out[end.values] = 1 - models[2].predict_proba(feats(s, 2))[:, 1]
    live = np.isnan(out)
    if live.any():
        dl = d[live].copy(); dl['balls_left'] = dl.balls_left.clip(lower=1)
        out[live] = models[i].predict_proba(feats(dl, i))[:, 1]
    return out

def ov2balls(o): return np.where(pd.isna(o), 120, np.floor(o) * 6 + np.round((o % 1) * 10))
b['max_balls2'] = ov2balls(b.second_inn_revised_overs)
reg = b[b.innings <= 2].copy()
lev = np.zeros(len(reg)); wp_exp = {}
for i in (1, 2):
    ii = (reg.innings == i).values; d = reg[ii]
    base = d.wp.values; acc = np.zeros(len(d))
    for oc, (r, lg, w) in OUTCOMES.items():
        h = d.copy()
        h['runs_before'] = h.runs_before + r; h['legal_before'] = h.legal_before + lg
        h['balls_left'] = h.balls_left - lg; h['wkts_in_hand'] = h.wkts_in_hand - w
        if i == 2: h['runs_needed'] = h.runs_needed - r
        p = d[['phase']].merge(mix[(mix.innings == i) & (mix.oc == oc)], on='phase', how='left').p.fillna(0).values
        acc += p * np.abs(wp_state(h, i) - base)
    lev[ii] = acc
reg['leverage_raw'] = lev
reg['LI'] = reg.leverage_raw / reg.leverage_raw.mean()          # 1.0 = average ball

# actual WP after each ball (for 'actual swing' check and later WPA)
reg['wp_after'] = reg.groupby(['match_id', 'innings']).wp.shift(-1)
last = reg.wp_after.isna()
for i in (1, 2):
    k = last & (reg.innings == i)
    h = reg[k].copy()
    h['runs_before'] += h.tot; h['legal_before'] += h.legal.astype(int); h['balls_left'] -= h.legal.astype(int)
    h['wkts_in_hand'] -= h.is_wicket.astype(int)
    if i == 2:
        h['runs_needed'] -= h.tot
        reg.loc[k, 'wp_after'] = np.where(reg.loc[k, 'trainable'], reg.loc[k, 'won'], wp_state(h, 2))  # real result if known
    else:
        h['balls_left'] = 0; reg.loc[k, 'wp_after'] = wp_state(h, 1)
reg['wp_swing'] = (reg.wp_after - reg.wp).abs()

# ---------- pressure score & within-phase cut ----------
reg['ko_w'] = np.where(reg.knock_out == 1, KO_W, 1.0)
reg['p_score'] = reg.LI * reg.ko_w
for t in (0.10, 0.15, 0.20):
    cut = reg.groupby(['phase', 'innings']).p_score.transform(lambda s: s.quantile(1 - t))
    reg[f'pressure_{int(t*100)}'] = reg.p_score >= cut
reg['pressure'] = reg[f'pressure_{int(TOP*100)}']
reg['phase_w'] = reg.phase.map(PHASE_W)
so = b[b.innings > 2].copy(); so['pressure'] = True; so['pressure_10'] = so['pressure_15'] = so['pressure_20'] = True; so['phase'] = 'Super over'; so['phase_w'] = 1.25; so['LI'] = np.nan
allb = pd.concat([reg, so], ignore_index=True)

# ---------- build-up situations ----------
allb = allb.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
allb['is_bdry'] = allb.is_four | allb.is_six
allb['is_dot'] = allb.legal & (allb.tot == 0) & ~allb.is_wicket
# striker's consecutive dots before this ball (resets on any run / boundary)
def run_len(s):
    out, c = [], 0
    for v in s: out.append(c); c = c + 1 if v else 0
    return out
allb['striker_dots_before'] = allb.groupby(['match_id', 'innings', 'striker_id'], sort=False).is_dot.transform(run_len)
grp = allb.groupby(['match_id', 'innings'], sort=False)
allb['balls_since_bdry'] = grp.is_bdry.transform(lambda s: run_len(~s.values))
allb['striker_faced_before'] = allb.groupby(['match_id', 'innings', 'striker_id'], sort=False).legal.cumsum() - allb.legal.astype(int)
allb['wkt_prev2'] = grp.is_wicket.transform(lambda s: s.shift(1, fill_value=False) | s.shift(2, fill_value=False))
allb['prev_ball_bdry'] = grp.is_bdry.shift(1, fill_value=False) & (grp.bowler_id.shift(1) == allb.bowler_id)
allb['over_no'] = allb.legal_before // 6
ob = allb.groupby(['match_id', 'innings', 'over_no'], sort=False)
allb['bdry_in_over_before'] = ob.is_bdry.cumsum() - allb.is_bdry.astype(int)
ext_prev = grp.extras_type.shift(1)
allb['prev_wide_nb'] = ext_prev.isin(['wide', 'Overstep', 'other']) & (grp.bowler_id.shift(1) == allb.bowler_id)
allb['free_hit'] = ext_prev.isin(['Overstep', 'other'])
ov = allb.groupby(['match_id', 'innings', 'over_no', 'bowler_id'], sort=False).tot.sum().rename('ov_runs').reset_index()
ov['prev_over_runs'] = ov.groupby(['match_id', 'innings', 'bowler_id']).ov_runs.shift(1)
allb = allb.merge(ov[['match_id', 'innings', 'over_no', 'bowler_id', 'prev_over_runs']], on=['match_id', 'innings', 'over_no', 'bowler_id'], how='left')

S_BAT = {'3+ dots in a row (striker)': allb.striker_dots_before >= 3,
         'No boundary for 12+ balls': allb.balls_since_bdry >= 12,
         'Wicket in last 2 balls': allb.wkt_prev2,
         'New batter (<5 balls faced)': allb.striker_faced_before < 5}
S_BOWL = {'Hit for boundary last ball': allb.prev_ball_bdry,
          '2+ boundaries already this over': allb.bdry_in_over_before >= 2,
          'Previous over went for 12+': allb.prev_over_runs >= 12,
          'Set batter (15+ balls faced)': allb.striker_faced_before >= 15,
          'Just bowled wide/no-ball': allb.prev_wide_nb,
          'Free hit': allb.free_hit}
for k, v in {**S_BAT, **S_BOWL}.items(): allb['sit_' + k] = v.values
allb.to_pickle(OUT + 'balls_pressure.pkl')

# ================= CHECKS =================
R = {}
r = allb[allb.innings <= 2]
legal = r[r.legal]
def outcome_tbl(mask):
    x = legal[mask]
    return dict(balls=len(x), runs_per_ball=round(x.tot.mean(), 3), dot_pct=round(x.is_dot.mean() * 100, 1),
                bdry_pct=round(x.is_bdry.mean() * 100, 1), wkt_pct=round(x.is_wicket.mean() * 100, 2),
                avg_actual_wp_swing=round(x.wp_swing.mean() * 100, 2))
R['pressure_vs_ordinary'] = {'Pressure': outcome_tbl(legal.pressure), 'Ordinary': outcome_tbl(~legal.pressure)}
R['swing_ratio'] = round(legal[legal.pressure].wp_swing.mean() / legal[~legal.pressure].wp_swing.mean(), 2)
R['LI_by_phase'] = r.groupby(['innings', 'phase']).LI.agg(['mean', 'median']).round(2).reset_index().to_dict('records')
R['LI_by_phase_all'] = r.groupby('phase').LI.agg(['mean', 'median', lambda s: s.quantile(.9)]).round(2).rename(columns={'<lambda_0>': 'p90'}).to_dict('index')
R['share_pressure_by_innings'] = r.groupby(['phase', 'innings']).pressure.mean().round(3).unstack().to_dict('index')
R['threshold_options'] = {t: dict(pressure_balls=int(r[f'pressure_{t}'].sum()),
                                  swing_ratio=round(r[r[f'pressure_{t}']].wp_swing.mean() / r[~r[f'pressure_{t}']].wp_swing.mean(), 2))
                          for t in (10, 15, 20)}
R['knockout_pressure_share'] = round(r[r.knock_out == 1].pressure.mean(), 3)

# situation tests: effect vs same innings x phase x wickets-bucket baseline
legal = legal.assign(wb=pd.cut(legal.wkts_in_hand, [-1, 3, 6, 10]))
strata = ['innings', 'phase', 'wb']
def sit_effect(mask):
    x = legal.assign(s=mask[legal.index].values)
    g = x.groupby(strata + ['s'], observed=True).agg(rpb=('tot', 'mean'), wk=('is_wicket', 'mean'), n=('tot', 'size')).unstack('s')
    g = g.dropna()
    w = g[('n', True)]
    d_r = ((g[('rpb', True)] - g[('rpb', False)]) * w).sum() / w.sum()
    d_w = ((g[('wk', True)] - g[('wk', False)]) * w).sum() / w.sum()
    base_r = (g[('rpb', False)] * w).sum() / w.sum()
    return dict(balls=int(w.sum()), runs_per_ball_diff=round(d_r, 3), runs_pct_change=round(d_r / base_r * 100, 1),
                wicket_rate_diff_pctpts=round(d_w * 100, 2))
R['batter_situations'] = {k: sit_effect(allb['sit_' + k]) for k in S_BAT}
R['bowler_situations'] = {k: sit_effect(allb['sit_' + k]) for k in S_BOWL}
R['overlap_situation_and_leverage'] = {k: round(allb.loc[allb['sit_' + k] & (allb.innings <= 2), 'pressure'].mean(), 3) for k in {**S_BAT, **S_BOWL}}
json.dump(R, open(OUT + 'pressure_checks.json', 'w'), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
