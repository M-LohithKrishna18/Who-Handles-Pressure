"""Stage 7: reliability.
 (a) match bootstrap (200x): how often each eligible player is in the top 10, mean rank
 (b) split-half reliability of raw pressure RAA/100 (random match halves, 50 repeats, Spearman-Brown)
 (c) sensitivity: pressure threshold 10%/20%, phase weights 1.0, IPL-only, T20I-only -> top-10 overlap
 (d) is pressure skill different from general skill? corr(pressure RAA, ordinary RAA)"""
import os
import pandas as pd, numpy as np, json
from s4_ratings import build, rate, OUT, MIN_BALLS, MIN_INN
rng = np.random.default_rng(7)
ex = pd.read_csv(OUT + 'exclusions.csv').set_index('pid')
excluded = ex.excl_pak_ban.notna() | ex.excl_inactive | ex.excl_retained.notna() | ex.excl_banned
params = json.load(open(OUT + 'rating_params.json'))
R = {}

def top10(t):
    el = t[t.eligible_sample & ~t.index.map(excluded).fillna(False).astype(bool)]
    return list(el.sort_values('rating', ascending=False).index[:10])

bat, bowl = build()
for role, df in (('bat', bat), ('bowl', bowl)):
    main = pd.read_csv(OUT + f'{role}_pool.csv', index_col=0)          # candidate pool (stage 6)
    prior = main.prior
    main_top = list(main.index[:10]); names = main.name.to_dict()
    k = params[role]['k_balls']
    p = df[df.pressure]
    agg = p.groupby(['pid', 'match_id']).apply(lambda d: pd.Series({'w': d.w.sum(), 'w2': (d.w ** 2).sum(), 'r': (d.raa * d.w).sum(), 'n': len(d)}), include_groups=False).reset_index()
    matches = agg.match_id.unique()
    elig_ids = main.index
    # (a) bootstrap
    in_top, ranks = pd.Series(0, index=elig_ids), {i: [] for i in elig_ids}
    B = 200
    for _ in range(B):
        cnt = pd.Series(rng.multinomial(len(matches), np.ones(len(matches)) / len(matches)), index=matches)
        a = agg.assign(m=agg.match_id.map(cnt)); a = a[a.m > 0]
        g = a.assign(w=a.w * a.m, w2=a.w2 * a.m, r=a.r * a.m, n=a.n * a.m, i=a.m).groupby('pid')[['w', 'w2', 'r', 'n', 'i']].sum()
        g = g[(g.n >= MIN_BALLS) & (g.i >= MIN_INN) & g.index.isin(elig_ids)]
        ne = g.w ** 2 / g.w2
        g['rating'] = (ne * (g.r / g.w * 100) + k * prior.reindex(g.index)) / (ne + k)
        rk = g.rating.rank(ascending=False)
        in_top[rk[rk <= 10].index] += 1
        for i, v in rk.items(): ranks[i].append(v)
    bs = pd.DataFrame({'name': main.name, 'rating': main.rating.round(2), 'top10_pct': (in_top / B * 100).round(0),
                       'median_rank': {i: np.median(v) if v else np.nan for i, v in ranks.items()}})
    bs = bs.sort_values('top10_pct', ascending=False)
    bs.to_csv(OUT + f'{role}_bootstrap.csv')
    # (b) split-half
    cors = []
    for _ in range(50):
        h = pd.Series(rng.random(len(matches)) < 0.5, index=matches)
        a = agg.assign(h=agg.match_id.map(h))
        g = a.groupby(['pid', 'h'])[['w', 'r', 'n']].sum().unstack('h')
        g = g[(g[('n', True)] >= 30) & (g[('n', False)] >= 30)]
        c = np.corrcoef(g[('r', True)] / g[('w', True)], g[('r', False)] / g[('w', False)])[0, 1]
        cors.append(c)
    c = float(np.mean(cors)); sb = 2 * c / (1 + c) if c > -1 else np.nan
    # (c) sensitivity
    sens = {}
    for lab, kw in [('threshold 10%', dict(pcol='pressure_10')), ('threshold 20%', dict(pcol='pressure_20')),
                    ('phase weights 1.0', dict(phase_w={'Powerplay': 1, 'Middle': 1, 'Death': 1})),
                    ('IPL only', dict(comp='IPL')), ('T20I only', dict(comp='T20I'))]:
        vb, vw = build(**kw); t, _ = rate(vb if role == 'bat' else vw)
        tt = top10(t)
        sens[lab] = dict(overlap_with_main=len(set(tt) & set(main_top)), top10=[names.get(i, t.name.get(i)) for i in tt],
                         n_eligible=int((t.eligible_sample & ~t.index.map(excluded).fillna(False).astype(bool)).sum()))
    # (d) pressure vs ordinary skill
    allr = pd.read_csv(OUT + f'{role}_ratings_all.csv', index_col=0); s = allr[allr.eligible_sample]
    R[role] = dict(split_half_corr=round(c, 3), spearman_brown_full=round(sb, 3), players_in_split=int(len(g)),
                   bootstrap=bs.head(15).to_dict('index'), sensitivity=sens,
                   corr_pressure_vs_ordinary=round(s.raa100_raw.corr(s.ord_raa100), 3),
                   corr_rating_vs_wpa=round(s.rating.corr(s.wpa100), 3))
    print(f'\n===== {role.upper()}  split-half r={c:.2f} (full-sample reliability {sb:.2f}) | pressure-vs-ordinary corr {R[role]["corr_pressure_vs_ordinary"]} | rating-vs-WPA corr {R[role]["corr_rating_vs_wpa"]}')
    print(bs.head(15).to_string())
    for k2, v in sens.items(): print(f'  {k2:18s} overlap {v["overlap_with_main"]}/10 (eligible {v["n_eligible"]})')
json.dump(R, open(OUT + 'reliability.json', 'w'), indent=1, default=str)
