"""Stage 1: ball-level match state + win-probability model (one per innings).
Train 2022-2025, test 2026 (split by match). LightGBM w/ monotone constraints vs logistic regression."""
import os
import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_clean.pkl'); m = pd.read_pickle(OUT + 'matches_clean.pkl')
m['match_date'] = pd.to_datetime(m.match_date)
b = b[~b.super_over].copy()
b['tot'] = b.runs + b.extras1_runs + b.extras_2_runs
b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

def ov2balls(o):
    return np.where(pd.isna(o), 120, np.floor(o) * 6 + np.round((o % 1) * 10))

# ---- innings totals, venue par (leave-one-match-out, shrunk), Elo ----
inn = b.groupby(['match_id', 'innings']).agg(total=('tot', 'sum'), bat=('batting_team_id', 'first'),
                                              bowl=('bowling_team_id', 'first')).reset_index()
m = m.merge(inn[inn.innings == 1][['match_id', 'total']].rename(columns={'total': 'inn1_total'}), on='match_id', how='left')
full1 = m[m.first_inn_revised_overs.isna() & ~m.no_result & m.inn1_total.notna()]
grp = full1.groupby(['comp_group', 'year']).inn1_total.mean()
vs = full1.groupby('venue_id').inn1_total.agg(['sum', 'count'])
K = 5
def par(r):
    base = grp.get((r.comp_group, r.year), full1.inn1_total.mean())
    s, n = (vs.loc[r.venue_id] if r.venue_id in vs.index else (0, 0))
    if r.match_id in full1.match_id.values: s, n = s - r.inn1_total, n - 1   # leave own match out
    return (s + K * base) / (n + K)
m['par'] = m.apply(par, axis=1)

elo, pre = {}, {}
for r in m.sort_values('match_date').itertuples():
    h, a = elo.get(r.home_team_id, 1500), elo.get(r.away_team_id, 1500)
    pre[r.match_id] = (h, a)
    if pd.isna(r.win_team_id) or r.no_result: continue
    eh = 1 / (1 + 10 ** ((a - h) / 400)); sh = 1.0 if r.win_team_id == r.home_team_id else 0.0
    elo[r.home_team_id] = h + 20 * (sh - eh); elo[r.away_team_id] = a - 20 * (sh - eh)
m['elo_home'] = m.match_id.map(lambda x: pre[x][0]); m['elo_away'] = m.match_id.map(lambda x: pre[x][1])
m['impact_era'] = ((m.comp_group == 'IPL') & (m.year >= 2023)).astype(int)

# ---- per-ball state BEFORE the delivery ----
g = b.groupby(['match_id', 'innings'])
b['runs_before'] = g.tot.cumsum() - b.tot
b['wkts_before'] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
b['legal_before'] = g.legal.cumsum() - b.legal.astype(int)
b = b.merge(m[['match_id', 'comp_group', 'year', 'par', 'elo_home', 'elo_away', 'home_team_id', 'impact_era',
               'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target', 'inn1_total',
               'win_team_id', 'no_result', 'tied', 'knock_out', 'match_date']], on='match_id', how='left')
b['max_balls'] = np.where(b.innings == 1, ov2balls(b.first_inn_revised_overs), ov2balls(b.second_inn_revised_overs))
b['balls_left'] = (b.max_balls - b.legal_before).clip(lower=1)
b['wkts_in_hand'] = 10 - b.wkts_before
bat_elo = np.where(b.batting_team_id == b.home_team_id, b.elo_home, b.elo_away)
bowl_elo = np.where(b.batting_team_id == b.home_team_id, b.elo_away, b.elo_home)
b['elo_diff'] = bat_elo - bowl_elo
b['crr'] = b.runs_before / np.maximum(b.legal_before, 1) * 6
b['par_scaled'] = b.par * b.max_balls / 120
b['proj_vs_par'] = b.runs_before + b.crr / 6 * b.balls_left - b.par_scaled
b['target'] = np.where(b.revised_target.notna(), b.revised_target, b.inn1_total + 1)
b['runs_needed'] = b.target - b.runs_before
b['rrr'] = b.runs_needed / b.balls_left * 6
b['is_ipl'] = (b.comp_group == 'IPL').astype(int); b['is_ateam'] = (b.comp_group == 'A-team').astype(int)
b['won'] = (b.batting_team_id == b.win_team_id).astype(float)
b['trainable'] = ~b.no_result & ~b.tied & b.win_team_id.notna()
b['is_test'] = b.year >= 2026

F = {1: ['balls_left', 'wkts_in_hand', 'runs_before', 'crr', 'par_scaled', 'proj_vs_par', 'elo_diff',
         'is_ipl', 'is_ateam', 'impact_era', 'year', 'max_balls'],
     2: ['balls_left', 'wkts_in_hand', 'runs_needed', 'rrr', 'target', 'par_scaled', 'elo_diff',
         'is_ipl', 'is_ateam', 'impact_era', 'year', 'max_balls']}
MONO = {1: [1, 1, 1, 1, -1, 1, 1, 0, 0, 0, 0, 0],
        2: [1, 1, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0]}
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=400, n_estimators=600,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)

res, models = {}, {}
b['wp'] = np.nan
for i in (1, 2):
    d = b[(b.innings == i) & b.trainable]
    tr, te = d[~d.is_test], d[d.is_test]
    gbm = lgb.LGBMClassifier(**P, monotone_constraints=MONO[i]).fit(tr[F[i]], tr.won)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[F[i]], tr.won)
    r = {}
    for name, mod in [('LightGBM', gbm), ('Logistic', lr)]:
        pr = mod.predict_proba(te[F[i]])[:, 1]
        r[name] = dict(logloss=round(log_loss(te.won, pr), 4), brier=round(brier_score_loss(te.won, pr), 4),
                       auc=round(roc_auc_score(te.won, pr), 4))
        if name == 'LightGBM':
            cal = pd.DataFrame({'p': pr, 'y': te.won.values})
            cal['bin'] = pd.cut(cal.p, np.linspace(0, 1, 11))
            r['calibration'] = cal.groupby('bin', observed=True).agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3).reset_index().astype({'bin': str}).to_dict('records')
    r['test_matches'] = int(te.match_id.nunique()); r['train_matches'] = int(tr.match_id.nunique())
    r['importance'] = dict(sorted(zip(F[i], gbm.booster_.feature_importance('gain').round(0).tolist()), key=lambda x: -x[1]))
    res[i] = r
    # final model on all trainable data -> predict every ball of this innings (incl. no-result)
    models[i] = lgb.LGBMClassifier(**P, monotone_constraints=MONO[i]).fit(d[F[i]], d.won)
    idx = b.innings == i
    b.loc[idx, 'wp'] = models[i].predict_proba(b.loc[idx, F[i]])[:, 1]

json.dump(res, open(OUT + 'wp_eval.json', 'w'), indent=1, default=str)
b.to_pickle(OUT + 'balls_state.pkl')
import pickle; pickle.dump({'models': models, 'F': F}, open(OUT + 'wp_models.pkl', 'wb'))
for i in (1, 2):
    print(f'--- Innings {i}: train {res[i]["train_matches"]} matches, test(2026) {res[i]["test_matches"]}')
    for n in ('LightGBM', 'Logistic'): print(n, res[i][n])
    print('calibration (pred vs actual):', [(c['pred'], c['actual'], c['n']) for c in res[i]['calibration']])
    print('importance:', res[i]['importance'])
