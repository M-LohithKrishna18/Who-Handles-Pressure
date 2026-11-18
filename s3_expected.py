"""Stage 3: context baselines (no player identity).
  xR_bat  : expected runs off the bat for an average batter (non-wide balls)
  xOut_bat: P(striker dismissed)                            (non-wide balls)
  xR_bowl : expected runs charged to an average bowler       (all balls)
  xW_bowl : P(bowler-credited wicket)                        (all balls)
All predictions are out-of-fold (5-fold, grouped by match) so a ball never predicts itself.
Wicket run-value: drop in expected remaining-innings runs when a wicket falls (DLS-style table)."""
import os
import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
OUT = os.environ.get('OUT_DIR', 'out/')
b = pd.read_pickle(OUT + 'balls_pressure.pkl')
b = b[b.innings <= 2].copy()          # super overs rated later as add-on (no meaningful baseline)

NB = ['wide', 'Overstep', 'other']
b['bat_runs'] = b.runs
b['bowl_runs'] = b.runs + np.where(b.extras_type.isin(NB), b.extras1_runs + b.extras_2_runs, 0)
b['striker_out'] = b.is_wicket & (b.dismissed_player_id == b.striker_id)
b['bowler_wkt'] = b.is_wicket & ~b.dismissal_type.isin(['RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'])
b['faced'] = b.extras_type != 'wide'

b['over'] = (b.legal_before // 6 + 1).clip(upper=20)
b['rrr_c'] = np.where(b.innings == 2, (b.runs_needed / b.balls_left.clip(lower=1) * 6).clip(-6, 36), np.nan)
b['runs_vs_par'] = b.runs_before - b.par * (b.legal_before / 120)
b['lhb'] = (b.batting_type == 'LHB').astype(int)
b['pace'] = (b.pace_or_spin == 'Pace').astype(int)
bt = b.bowling_type.fillna('NA').str.lower()
b['bowl_style'] = np.select([bt.str.contains('fast|medium'), bt.str.contains('legbreak|googly|wrist|chinaman'),
                             bt.str.contains('left-arm orthodox|slow left'), bt.str.contains('offbreak|off')],
                            [0, 1, 2, 3], 4)
b['left_arm'] = bt.str.contains('left').astype(int)
b['same_side'] = ((b.lhb == 1) == (b.left_arm == 1)).astype(int)
for c in ['comp_group']: b[c + '_c'] = b[c].astype('category').cat.codes
b['ko'] = (b.knock_out == 1).astype(int)
b['free_hit_i'] = b.free_hit.astype(int)

F = ['innings', 'over', 'legal_before', 'balls_left', 'wkts_in_hand', 'rrr_c', 'runs_vs_par', 'par', 'year',
     'comp_group_c', 'impact_era', 'lhb', 'pace', 'bowl_style', 'left_arm', 'same_side', 'ko', 'elo_diff',
     'striker_dots_before', 'balls_since_bdry', 'striker_faced_before', 'wkt_prev2', 'bdry_in_over_before',
     'prev_over_runs', 'free_hit_i']
b['wkt_prev2'] = b.wkt_prev2.astype(int)
P = dict(learning_rate=0.05, num_leaves=31, min_child_samples=300, n_estimators=400, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)

PW = dict(P, learning_rate=0.03, num_leaves=7, min_child_samples=2000, n_estimators=250, reg_lambda=20)

def oof(mask, y, kind):
    cols = F + (['batting_order'] if 'bowl' in y else [])   # who is batting matters for bowlers (tailenders)
    d = b[mask]; X = d[cols]; yv = d[y].astype(float); pred = np.zeros(len(d))
    for a, z in GroupKFold(5).split(X, yv, d.match_id):
        mdl = (lgb.LGBMRegressor(**P, objective='poisson') if kind == 'reg' else lgb.LGBMClassifier(**PW))
        mdl.fit(X.iloc[a], yv.iloc[a])
        pred[z] = mdl.predict(X.iloc[z]) if kind == 'reg' else mdl.predict_proba(X.iloc[z])[:, 1]
    # naive baseline: innings x over average
    naive = d.groupby(['innings', 'over'])[y].transform('mean').astype(float)
    if kind == 'reg':
        ev = dict(rmse_model=round(mean_squared_error(yv, pred) ** .5, 4), rmse_naive=round(mean_squared_error(yv, naive) ** .5, 4),
                  mean_actual=round(yv.mean(), 4), mean_pred=round(pred.mean(), 4))
    else:
        ev = dict(logloss_model=round(log_loss(yv, pred), 4), logloss_naive=round(log_loss(yv, naive), 4),
                  auc_model=round(roc_auc_score(yv, pred), 4), auc_naive=round(roc_auc_score(yv, naive), 4),
                  mean_actual=round(yv.mean(), 4), mean_pred=round(pred.mean(), 4))
    b.loc[mask, 'x_' + y] = pred
    return ev

EV = {}
EV['xR_bat'] = oof(b.faced, 'bat_runs', 'reg')
EV['xOut_bat'] = oof(b.faced, 'striker_out', 'clf')
EV['xR_bowl'] = oof(b.index == b.index, 'bowl_runs', 'reg')  # uses batting_order
EV['xW_bowl'] = oof(b.index == b.index, 'bowler_wkt', 'clf')

# ---- wicket run value: expected remaining runs R(balls_left, wkts_in_hand), first innings, full 20 overs ----
f = b[(b.innings == 1) & b.first_inn_revised_overs.isna()].copy()
tot = f.groupby('match_id').tot.sum()
f['remaining'] = f.match_id.map(tot) - f.runs_before
R = f.groupby([pd.cut(f.balls_left, np.arange(0, 126, 6)), 'wkts_in_hand'], observed=True).remaining.mean()
Rt = R.unstack().interpolate(axis=1, limit_direction='both').ffill().bfill()
def rem(bl, w):
    i = np.clip((np.asarray(bl) - 1) // 6, 0, 19).astype(int); w = np.asarray(w).astype(int)
    v = Rt.values[i, np.clip(w, 1, 10) - 1]
    return np.where(w < 1, 0.0, v)
bl, w = b.balls_left.clip(lower=1), b.wkts_in_hand.clip(lower=1)
b['wkt_value'] = np.clip(rem(bl, w) - rem((bl - 1).clip(lower=1), w - 1), 2, None)
EV['wicket_value_runs_by_phase'] = b.groupby('phase').wkt_value.mean().round(1).to_dict()

b.to_pickle(OUT + 'balls_expected.pkl')
json.dump(EV, open(OUT + 'expected_eval.json', 'w'), indent=1)
print(json.dumps(EV, indent=1))
