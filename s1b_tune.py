"""Stage 1b: tune WP models with match-grouped CV on 2022-25 only; final check on 2026."""
import os
import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_state.pkl')
# engineered features (generalisable, no match fingerprints)
b['frac_left'] = b.balls_left / b.max_balls
b['log_bl'] = np.log1p(b.balls_left)
b['rn_per_wkt'] = b.runs_needed / b.wkts_in_hand
b['rrr_c'] = b.rrr.clip(-6, 36)
b['req_minus_par_rate'] = b.rrr_c - b.par / 20          # chase rate vs venue scoring rate
b['res_left'] = b.frac_left * (b.wkts_in_hand / 10) ** 0.7  # simple DLS-style resources left
b['need_per_res'] = (b.runs_needed / np.maximum(b.res_left * b.par, 1)).clip(-1, 3)
b['proj_c'] = b.proj_vs_par.clip(-120, 120)
b['proj_per_ball'] = b.proj_vs_par / b.max_balls
b['elo_p'] = 1 / (1 + 10 ** (-b.elo_diff / 400))
b['wkts_x_frac'] = b.wkts_in_hand * b.frac_left

FS = {
 1: {'gbm': ['frac_left', 'wkts_in_hand', 'proj_c', 'res_left', 'elo_diff', 'is_ipl', 'impact_era'],
     'mono': [0, 1, 1, 1, 1, 0, 0],
     'lr': ['proj_c', 'proj_per_ball', 'wkts_in_hand', 'res_left', 'frac_left', 'elo_p', 'is_ipl', 'impact_era', 'is_ateam']},
 2: {'gbm': ['balls_left', 'wkts_in_hand', 'runs_needed', 'rrr_c', 'req_minus_par_rate', 'elo_diff', 'is_ipl', 'impact_era'],
     'mono': [1, 1, -1, -1, -1, 1, 0, 0],
     'lr': ['rrr_c', 'req_minus_par_rate', 'need_per_res', 'res_left', 'wkts_in_hand', 'log_bl', 'rn_per_wkt',
            'wkts_x_frac', 'elo_p', 'is_ipl', 'impact_era', 'is_ateam']}}
GRID = [dict(num_leaves=7, min_child_samples=2000, n_estimators=300, learning_rate=0.03),
        dict(num_leaves=7, min_child_samples=5000, n_estimators=400, learning_rate=0.03),
        dict(num_leaves=4, min_child_samples=3000, n_estimators=500, learning_rate=0.03)]
BASE = dict(objective='binary', subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=20, verbose=-1)

def metrics(y, p): return dict(logloss=round(log_loss(y, p), 4), brier=round(brier_score_loss(y, p), 4), auc=round(roc_auc_score(y, p), 4))
def calib(y, p):
    c = pd.DataFrame({'p': p, 'y': np.asarray(y)}); c['bin'] = pd.cut(c.p, np.linspace(0, 1, 11))
    t = c.groupby('bin', observed=True).agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size'))
    return t, float((t.n * (t.pred - t.actual).abs()).sum() / t.n.sum())   # expected calibration error

out = {}
for i in (1, 2):
    d = b[(b.innings == i) & b.trainable]; tr, te = d[~d.is_test], d[d.is_test]
    gkf = GroupKFold(5); cv = {}
    def cvscore(make, cols):
        ps = np.zeros(len(tr))
        for a, z in gkf.split(tr, tr.won, tr.match_id):
            ps[z] = make().fit(tr.iloc[a][cols], tr.iloc[a].won).predict_proba(tr.iloc[z][cols])[:, 1]
        return log_loss(tr.won, ps)
    for k, g in enumerate(GRID):
        cv[f'gbm{k}'] = cvscore(lambda: lgb.LGBMClassifier(**BASE, **g, monotone_constraints=FS[i]['mono']), FS[i]['gbm'])
    for C in (0.1, 1.0):
        cv[f'lr_C{C}'] = cvscore(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), FS[i]['lr'])
    best_g = min([k for k in cv if k.startswith('gbm')], key=cv.get); best_l = min([k for k in cv if k.startswith('lr')], key=cv.get)
    g = GRID[int(best_g[3:])]; C = float(best_l.split('C')[1])
    gbm = lgb.LGBMClassifier(**BASE, **g, monotone_constraints=FS[i]['mono']).fit(tr[FS[i]['gbm']], tr.won)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(tr[FS[i]['lr']], tr.won)
    pg, pl = gbm.predict_proba(te[FS[i]['gbm']])[:, 1], lr.predict_proba(te[FS[i]['lr']])[:, 1]
    pe = (pg + pl) / 2
    r = {'cv_logloss_2022_25': {k: round(v, 4) for k, v in cv.items()}, 'chosen': [best_g, best_l]}
    for name, p in [('LightGBM', pg), ('Logistic', pl), ('Average of both', pe)]:
        t, ece = calib(te.won, p); r[name] = {**metrics(te.won, p), 'ECE': round(ece, 3)}
        r[name + '_calib'] = t.round(3).reset_index().astype({'bin': str}).to_dict('records')
    out[i] = r
    print(f'=== Innings {i}  CV(2022-25) logloss: {r["cv_logloss_2022_25"]}')
    for n in ('LightGBM', 'Logistic', 'Average of both'): print(f'  2026 test  {n:16s}', r[n])
json.dump(out, open(OUT + 'wp_eval_tuned.json', 'w'), indent=1, default=str)
b.to_pickle(OUT + 'balls_state.pkl')
