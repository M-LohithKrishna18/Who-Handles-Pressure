"""Win-probability features, shared by training and by 'what-if' scoring (stage 2).
Input columns needed: runs_before, legal_before, wkts_in_hand, balls_left, max_balls, par,
elo_diff, is_ipl, is_ateam, impact_era, and (2nd innings) runs_needed."""
import os
import numpy as np, pandas as pd

def feats(d: pd.DataFrame, innings: int) -> pd.DataFrame:
    x = pd.DataFrame(index=d.index)
    fl = d.balls_left / d.max_balls
    w = d.wkts_in_hand
    res = fl * (w / 10) ** 0.7                       # DLS-style resources left (balls x wickets)
    x['frac_left'], x['wkts'], x['res_left'] = fl, w, res
    x['wkts_sq'] = w ** 2 / 10
    x['wkts_x_frac'] = w * fl                         # value of a wicket depends on balls left
    x['elo_p'] = 1 / (1 + 10 ** (-d.elo_diff / 400))  # team-strength expected result
    x['is_ipl'], x['is_ateam'], x['impact_era'] = d.is_ipl, d.is_ateam, d.impact_era
    if innings == 1:
        par = d.par * d.max_balls / 120
        # scoring rate blended with venue par rate (worth 18 balls) so early balls don't swing the projection
        K = 18
        rate = (d.runs_before + d.par / 120 * K) / (d.legal_before + K)
        proj = d.runs_before + rate * d.balls_left * (0.6 + 0.4 * w / 10)   # damped projection
        x['proj_vs_par'] = (proj - par).clip(-120, 120)
        x['runs_vs_par_sofar'] = (d.runs_before - par * (1 - fl)).clip(-100, 100)
        x['proj_x_frac'] = x.proj_vs_par * (1 - fl)   # projection gets more reliable later
    else:
        bl = d.balls_left
        rrr = (d.runs_needed / bl * 6).clip(-6, 36)
        x['log_bl'] = np.log1p(bl)
        x['rrr'] = rrr
        x['rrr_vs_par'] = rrr - d.par / 20
        x['need_per_res'] = (d.runs_needed / np.maximum(res * d.par, 1)).clip(-1, 3)
        x['rrr_x_wkts'] = rrr * w / 10
        x['rrr_x_logbl'] = rrr * x.log_bl
        x['need_per_wkt'] = (d.runs_needed / w).clip(-10, 80)
        x['need_x_frac'] = x.need_per_res * fl
    return x
