"""Stages 4-6: player pressure ratings and top-10 lists.
Per ball (pressure balls only for the rating):
  batter RAA = (runs - xRuns) - wicket_value * (out - xOut)
  bowler RAA = (xRunsConceded - runs conceded) + wicket_value * (wicket - xWicket)
Ball weight = phase weight (PP/death 1.25) x 0.5 if A-team competition or associate opposition.
Rating = empirical-Bayes shrunk pressure RAA per 100 balls (single number, in runs).
Supporting: WPA per 100 pressure balls, drop-off (pressure minus ordinary RAA/100), consistency
(share of pressure innings with positive RAA, shrunk), BASRA comparison."""
import os
import pandas as pd, numpy as np, json, sys
OUT = os.environ.get('OUT_DIR', 'out/')
MIN_BALLS, MIN_INN = 60, 8
DECAY = 0.75
USE_PRIOR = True      # shrink toward player's own (ordinary-ball) record, not toward 0          # recency: each season back counts x0.75 (2026=1, 2025=.75, 2024=.56, 2023=.42, 2022=.32)
PHASE_W = {'Powerplay': 1.25, 'Middle': 1.0, 'Death': 1.25}
ASSOC = {272, 283, 311, 312, 313, 282, 24, 26, 89, 315, 3886}  # UAE, HK, SGP, KUW, OMA, NEP, NED, SCO, CAN, USA, JPN

def build(phase_w=PHASE_W, pcol='pressure', comp=None, decay=DECAY):
    b = pd.read_pickle(OUT + 'balls_expected.pkl')
    if comp: b = b[b.comp_group == comp]
    b['pressure'] = b[pcol]
    b['w'] = b.phase.map(phase_w) * np.where((b.comp_group == 'A-team') | b.batting_team_id.isin(ASSOC) | b.bowling_team_id.isin(ASSOC), 0.5, 1.0)
    b['w'] = b.w * decay ** (2026 - b.year)
    b['wpa_bat'] = b.wp_after - b.wp                     # batting side win-prob change
    bat = b[b.faced].assign(raa=lambda d: (d.bat_runs - d.x_bat_runs) - d.wkt_value * (d.striker_out - d.x_striker_out),
                            pid=lambda d: d.striker_id, name=lambda d: d.striker_name, wpa=lambda d: d.wpa_bat)
    bowl = b.assign(raa=lambda d: (d.x_bowl_runs - d.bowl_runs) + d.wkt_value * (d.bowler_wkt - d.x_bowler_wkt),
                    pid=lambda d: d.bowler_id, name=lambda d: d.bowler_name, wpa=lambda d: -d.wpa_bat)
    return bat, bowl

def rate(df, so=None):
    p = df[df.pressure]; o = df[~df.pressure]
    g = p.groupby('pid')
    t = pd.DataFrame({'name': g.name.first(), 'p_balls': g.size(), 'p_inn': g.match_id.nunique(),
                      'w_sum': g.w.sum(), 'raa_sum': (p.raa * p.w).groupby(p.pid).sum(),
                      'wpa_sum': (p.wpa * p.w).groupby(p.pid).sum()})
    t['raa100_raw'] = t.raa_sum / t.w_sum * 100
    t['wpa100'] = t.wpa_sum / t.w_sum * 100
    # ---- ordinary-ball record (large sample) -> player's own prior ----
    og = o.groupby('pid')
    t['ord_raa100'] = (o.raa * o.w).groupby(o.pid).sum() / og.w.sum() * 100
    t['ord_balls'] = og.size()
    t['ord_neff'] = og.w.sum() ** 2 / (o.w ** 2).groupby(o.pid).sum()
    def eb_k(raw100, neff, resid100=None):
        """empirical-Bayes k = sigma2/tau2 from players with enough balls"""
        return None
    sig2_o = np.average((o.raa - np.average(o.raa, weights=o.w)) ** 2, weights=o.w)
    ok = t.ord_neff > 100
    tau2_o = max(np.average((t.ord_raa100[ok] / 100) ** 2, weights=t.ord_neff[ok]) - np.average(sig2_o / t.ord_neff[ok], weights=t.ord_neff[ok]), 1e-6)
    t['ord_shrunk'] = (t.ord_raa100 * t.ord_neff / (t.ord_neff + sig2_o / tau2_o)).fillna(0)
    # ---- pressure record, shrunk toward beta x own ordinary record (experience prior) ----
    t['w2_sum'] = (p.w ** 2).groupby(p.pid).sum()
    t['n_eff'] = t.w_sum ** 2 / t.w2_sum
    sig2 = np.average((p.raa - (p.raa * p.w).sum() / p.w.sum()) ** 2, weights=p.w)   # per-ball variance
    big = t[t.p_balls >= MIN_BALLS]
    x, y, wt = big.ord_shrunk / 100, big.raa100_raw / 100, big.n_eff
    beta = float(np.clip(np.sum(wt * x * y) / np.sum(wt * x * x), 0, 1.0)) if USE_PRIOR else 0.0
    t['prior'] = beta * t.ord_shrunk
    resid = (big.raa100_raw - beta * big.ord_shrunk) / 100
    tau2 = max(np.average(resid ** 2, weights=wt) - np.average(sig2 / wt, weights=wt), 1e-6)
    t['k'] = sig2 / tau2
    t['shrink'] = t.n_eff / (t.n_eff + t.k)
    t['rating'] = t.shrink * t.raa100_raw + (1 - t.shrink) * t.prior      # RAA per 100 pressure balls
    t['se'] = np.sqrt(1 / (t.n_eff / sig2 + 1 / tau2)) * 100
    t['ci_lo'], t['ci_hi'] = t.rating - 1.96 * t.se, t.rating + 1.96 * t.se
    # ordinary-ball RAA and drop-off
    t['dropoff'] = t.raa100_raw - t.ord_raa100
    # recent form: all balls in 2025-26 (no recency/pressure filter), RAA per 100
    r = df[df.year >= 2025]; rg = r.groupby('pid')
    t['form_2526_raa100'] = (r.raa * r.w).groupby(r.pid).sum() / rg.w.sum() * 100
    t['form_2526_balls'] = rg.size()                                       # + = better under pressure
    # consistency: share of pressure innings with positive RAA (shrunk toward league share, k=5)
    inn = p.groupby(['pid', 'match_id']).apply(lambda d: (d.raa * d.w).sum(), include_groups=False).rename('r').reset_index()
    league = (inn.r > 0).mean()
    ci = inn.groupby('pid').r.agg(pos=lambda s: (s > 0).sum(), n='size')
    t['consistency'] = (ci.pos + 5 * league) / (ci.n + 5)
    t['eligible_sample'] = (t.p_balls >= MIN_BALLS) & (t.p_inn >= MIN_INN)
    return t, dict(sigma2=sig2, tau2=tau2, k_balls=sig2 / tau2, league_pos_share=league, beta_prior=beta)

if __name__ == '__main__':
    ex = pd.read_csv(OUT + 'exclusions.csv').set_index('pid')
    ex['excluded'] = ex.excl_pak_ban.notna() | ex.excl_inactive | ex.excl_retained.notna() | ex.excl_banned
    ex['why'] = np.select([ex.excl_retained.notna(), ex.excl_pak_ban.notna(), ex.excl_banned, ex.excl_inactive],
                          ['retained (' + ex.excl_retained.fillna('') + ')', 'PAK/BAN', 'banned', 'inactive/retired'], '')
    bat, bowl = build()
    # super overs: add as pressure balls vs league super-over average
    so = pd.read_pickle(OUT + 'balls_pressure.pkl'); so = so[so.innings > 2]
    meta = {}
    for role, df in (('bat', bat), ('bowl', bowl)):
        t, prm = rate(df); meta[role] = prm
        t = t.join(ex[['excluded', 'why', 'last_year']])
        # BASRA (batters) / economy (bowlers) over ALL balls for comparison
        if role == 'bat':
            a = df.groupby('pid').agg(runs=('bat_runs', 'sum'), balls=('faced', 'sum'), outs=('striker_out', 'sum'))
            t['basra'] = (a.runs / a.outs.clip(lower=1) + a.runs / a.balls * 100)
            s = so.groupby('striker_id').runs.agg(['sum', 'size'])
        else:
            a = df.groupby('pid').agg(runs=('bowl_runs', 'sum'), legal=('legal', 'sum'), wk=('bowler_wkt', 'sum'))
            t['economy'] = a.runs / a.legal * 6; t['bowl_avg'] = a.runs / a.wk.clip(lower=1)
            s = so.groupby('bowler_id').runs.agg(['sum', 'size'])
        t['super_over_balls'] = s['size'].reindex(t.index).fillna(0).astype(int)
        t = t.sort_values('rating', ascending=False)
        t.to_csv(OUT + f'{role}_ratings_all.csv')
        el = t[t.eligible_sample & ~t.excluded.fillna(False)]
        el.to_csv(OUT + f'{role}_ratings_eligible.csv')
        cols = ['name', 'rating', 'ci_lo', 'ci_hi', 'p_balls', 'p_inn', 'wpa100', 'dropoff', 'consistency', 'ord_raa100']
        print(f'\n===== {role.upper()}: sample-eligible {t.eligible_sample.sum()} | after exclusions {len(el)} | k={prm["k_balls"]:.0f} balls')
        print(el[cols].head(15).round(2).to_string())
        print(f'--- reference: top 10 among ALL sample-eligible players (incl. excluded)')
        print(t[t.eligible_sample][['name', 'rating', 'p_balls', 'why']].head(10).round(2).to_string())
    json.dump(meta, open(OUT + 'rating_params.json', 'w'), indent=1)
