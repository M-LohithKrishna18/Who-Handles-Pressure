"""Stage 10: unrestricted top 10 (no availability filter) - the brief read literally.
Same score as s8 (0.8 x pressure rating + 0.2 x shrunk 2025-26 form). Rules kept: rated, active,
>= 50% of pressure balls from IPL or full-member T20Is. Removed: retention, PAK/BAN, form gate, bowler IPL rule.
Mohammad Nawaz left out: ICC annulled his results 7 Feb - 1 May 2026 (doping)."""
import os, pandas as pd
OUT = os.environ.get('OUT_DIR', 'out/')
ex = pd.read_csv(OUT + 'exclusions.csv').set_index('pid')
for role in ('bat', 'bowl'):
    t = pd.read_csv(OUT + f'{role}_ratings_all.csv', index_col=0)
    u = t[t.eligible_sample & (t.core_share >= 0.5)].copy()
    u = u[~u.index.map(ex.excl_inactive).fillna(False).astype(bool) & (u.name != 'Mohammad Nawaz')]
    n = u.form_2526_balls.fillna(0)
    u['final_score'] = 0.8 * u.rating + 0.2 * u.form_2526_raa100.fillna(0) * n / (n + 300)
    u = u.sort_values('final_score', ascending=False); u.insert(0, 'rank', range(1, len(u) + 1))
    u.to_csv(OUT + f'unrestricted_{role}_list.csv')
    print(f'\n== UNRESTRICTED {role.upper()}'); print(u.head(10)[['rank', 'name', 'final_score', 'rating', 'why']].round(1).to_string(index=False))
