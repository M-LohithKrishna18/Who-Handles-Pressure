"""Stage 0b: availability exclusions (rankings only; players stay in every model).
 - Retained by an IPL team for 2027 (data/retained_ipl2027.txt), matched to IDs (fuzzy + manual fixes)
 - Indian players count as active if they played since 2025 (domestic cricket is outside the data)
 - Manual, news-checked: retired (MS Dhoni, Mohit Sharma), not available at auction (Ben Duckett ban, R Ashwin retired from IPL)"""
import os, re, difflib, pandas as pd
D, OUT = os.environ.get('DATA_DIR', 'data/'), os.environ.get('OUT_DIR', 'out/')
ex = pd.read_csv(OUT + 'exclusions.csv')
norm = lambda s: re.sub(r'[^a-z ]', '', s.lower().replace('mohd', 'mohammed')).strip()
names = {norm(n): pid for n, pid in zip(ex.name, ex.pid)}
FIX = {'Tilak Varma': 'Namboori Tilak Varma', 'KL Rahul': 'Lokesh Rahul', 'Vijaykumar Vyshak': 'Vyshak Kumar',
       'Rasikh Dar': 'Rasikh Salam', 'Raj Angad Bawa': 'Raj Bawa', 'Smaran Ravichandran': 'Ravichandran Smaran'}
rows = []
for line in open('retained_ipl2027.txt'):
    team, ps = line.strip().split('|')
    for p in [x.strip() for x in ps.split(',')]:
        if p in FIX: pid, how = ex.loc[ex.name == FIX[p], 'pid'].iloc[0], 'manual'
        else:
            k = norm(p); pid, how = names.get(k), 'exact'
            if pid is None:
                c = difflib.get_close_matches(k, names, 1, 0.8)
                pid, how = (names[c[0]], 'fuzzy') if c else (None, 'not in data')
        rows.append((team, p, pid, how))
r = pd.DataFrame(rows, columns=['team', 'name', 'pid', 'match']); r.to_csv(OUT + 'retained_matched.csv', index=False)
ex['excl_retained'] = ex.pid.map(r.dropna(subset=['pid']).set_index('pid').team)
# Indian players: played for India / India A, or IPL-only
p11 = pd.read_csv(D + 'playing_eleven.csv'); p11 = p11[p11.is_played]
IPL = [1, 2, 3, 4, 5, 7, 8, 242, 3508, 3509]
intl = set(p11[~p11.team_id.isin(IPL)].player_id); ind = set(p11[p11.team_id.isin([18, 222])].player_id)
ex['indian'] = ex.pid.isin(ind) | ~ex.pid.isin(intl)
ex.loc[ex.indian & ex.excl_inactive & (ex.last_year >= 2025), 'excl_inactive'] = False
ex['excl_note'] = ''
for n, note in [('Mahendra Singh Dhoni', 'retired as a player'), ('Mohit Sharma', 'retired from all cricket 3 Dec 2025')]:
    ex.loc[ex.name == n, ['excl_inactive', 'excl_note']] = [True, note]
ex['excl_banned'] = False
for n, note in [('Ben Duckett', 'IPL ban after withdrawing from IPL 2026'), ('Ravichandran Ashwin', 'retired from IPL (Aug 2025)')]:
    ex.loc[ex.name == n, ['excl_banned', 'excl_note']] = [True, note]
ex.to_csv(OUT + 'exclusions.csv', index=False)
print('retained matched', r.pid.notna().sum(), '/', len(r))
