"""Stage 1c: final logistic win-probability model (with interaction terms).
Checks: (a) train 2022-25 -> test 2026; (b) match-grouped 5-fold CV on all seasons.
Final model fit on all completed matches; WP saved for every ball."""
import os
import pandas as pd, numpy as np, json, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from wp_features import feats
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_state.pkl')
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))

def report(y, p):
    c = pd.DataFrame({'p': p, 'y': np.asarray(y)}); c['bin'] = pd.cut(c.p, np.linspace(0, 1, 11))
    t = c.groupby('bin', observed=True).agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size'))
    ece = float((t.n * (t.pred - t.actual).abs()).sum() / t.n.sum())
    return dict(auc=round(roc_auc_score(y, p), 4), logloss=round(log_loss(y, p), 4),
                brier=round(brier_score_loss(y, p), 4), ECE=round(ece, 3)), t.round(3)

res, models = {}, {}
b['wp'] = np.nan
for i in (1, 2):
    d = b[(b.innings == i) & b.trainable]; X = feats(d, i); y = d.won
    te = d.is_test.values
    r_hold, _ = report(y[te], mk().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1])
    oof = np.zeros(len(d))
    for a, z in GroupKFold(5).split(X, y, d.match_id):
        oof[z] = mk().fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[z])[:, 1]
    r_cv, cal = report(y, oof)
    oof26 = report(y[te], oof[te])[0]
    models[i] = mk().fit(X, y)
    coefs = dict(zip(X.columns, models[i][-1].coef_[0].round(3)))
    res[i] = {'holdout_2026(train 22-25)': r_hold, 'grouped_cv_all_seasons': r_cv, 'cv_on_2026_matches': oof26,
              'calibration_cv': cal.reset_index().astype({'bin': str}).to_dict('records'), 'std_coefs': coefs,
              'matches': int(d.match_id.nunique())}
    idx = b.innings == i
    b.loc[idx, 'wp'] = models[i].predict_proba(feats(b[idx], i))[:, 1]
    print(f'=== Innings {i} ({res[i]["matches"]} completed matches)')
    for k in ('holdout_2026(train 22-25)', 'grouped_cv_all_seasons', 'cv_on_2026_matches'): print(f'  {k:28s}', res[i][k])
    print('  calibration (pred -> actual):', [(round(p, 2), round(a, 2)) for p, a in zip(cal.pred, cal.actual)])

json.dump(res, open(OUT + 'wp_eval_final.json', 'w'), indent=1, default=str)
pickle.dump(models, open(OUT + 'wp_final_models.pkl', 'wb'))
b.to_pickle(OUT + 'balls_state.pkl')

# sanity scenarios (IPL 2026, par 170, equal teams)
def scen(need, bl, w):
    s = pd.DataFrame([dict(runs_needed=need, balls_left=bl, wkts_in_hand=w, max_balls=120, par=170, elo_diff=0,
                           is_ipl=1, is_ateam=0, impact_era=1, runs_before=0, legal_before=120 - bl)])
    return round(models[2].predict_proba(feats(s, 2))[0, 1] * 100, 1)
print('\nExample chases (need, balls, wkts -> win %):')
for s in [(30, 12, 3), (24, 11, 3), (30, 11, 2), (29, 11, 3), (180, 120, 10), (60, 36, 6), (60, 36, 3), (10, 12, 5), (12, 6, 4)]:
    print(' ', s, scen(*s))
