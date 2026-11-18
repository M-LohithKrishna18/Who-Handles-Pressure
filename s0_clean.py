"""Stage 0: clean matches and balls, build exclusion lists for rankings."""
import os
import pandas as pd, numpy as np
U = os.environ.get('DATA_DIR', 'data/')
OUT = os.environ.get('OUT_DIR', 'out/')
m = pd.read_csv(U + 'matches.csv')
c = pd.read_csv(U + 'competitions.csv')
b = pd.read_csv(U + 'ball_by_ball.csv', low_memory=False)
p11 = pd.read_csv(U + 'playing_eleven.csv')
mp = pd.read_csv(U + 'competition_team_player_mapping.csv')

m = m.merge(c[['comp_id', 'comp_name', 'season']], on='comp_id')
m['comp_group'] = np.where(m.comp_name.str.contains('IPL'), 'IPL',
                  np.where(m.comp_id.isin([2670, 3289]), 'A-team', 'T20I'))
m['year'] = pd.to_datetime(m.match_date).dt.year

log = []
n0 = len(m)
abandoned = m.result.str.contains('Abandon', case=False, na=False)
log.append(f'Matches in file: {n0}')
log.append(f'Dropped abandoned: {abandoned.sum()}')
m = m[~abandoned]
noballs = ~m.match_id.isin(b.match_id)
log.append(f'Dropped with no ball-by-ball data (after abandoned): {noballs.sum()}')
m = m[~noballs]
m['no_result'] = m.result.str.contains('No Result', case=False, na=False)
m['tied'] = m.result.str.contains('Tied', case=False, na=False)
m['rain_revised'] = m.revised_target.notna()
log.append(f'Kept: {len(m)} matches  (no-result {m.no_result.sum()}, tied {m.tied.sum()}, '
           f'rain-revised {m.rain_revised.sum()}, knockout {m.knock_out.sum()})')
log.append('By competition: ' + str(m.comp_group.value_counts().to_dict()))

b = b[b.match_id.isin(m.match_id)].copy()
b['super_over'] = b.innings > 2
b['legal'] = ~b.extras_type.isin(['wide', 'Overstep', 'other'])
b['total_runs'] = b.runs + b.extras1_runs + b.extras_2_runs  # checked below
log.append(f'Balls kept: {len(b)}  (super-over balls {b.super_over.sum()}, flagged as max pressure)')

m.to_pickle(OUT + 'matches_clean.pkl')
b.to_pickle(OUT + 'balls_clean.pkl')

# ---------- exclusion lists (rankings only) ----------
names = (pd.concat([b[['striker_id', 'striker_name']].set_axis(['pid', 'name'], axis=1),
                    b[['bowler_id', 'bowler_name']].set_axis(['pid', 'name'], axis=1)])
         .drop_duplicates('pid'))
apps = p11[p11.is_played].merge(m[['match_id', 'year', 'comp_group']], on='match_id')
PAK_BAN = {25: 'Pakistan', 341: 'Pakistan A', 21: 'Bangladesh', 3093: 'Bangladesh A'}
nat = apps[apps.team_id.isin(PAK_BAN)].groupby('player_id').team_id.first().map(PAK_BAN)
last = apps.groupby('player_id').year.max()
inactive = last[last < 2026]

ex = names.copy()
ex['excl_pak_ban'] = ex.pid.map(nat)
ex['last_year'] = ex.pid.map(last)
ex['excl_inactive'] = ex.pid.isin(inactive.index)
ex.to_csv(OUT + 'exclusions.csv', index=False)
log.append(f'Players in ball data: {len(ex)}; Pakistan/Bangladesh: {ex.excl_pak_ban.notna().sum()}; '
           f'no 2026 match (inactive): {ex.excl_inactive.sum()}')
open(OUT + 'clean_log.txt', 'w').write('\n'.join(log))
print('\n'.join(log))
