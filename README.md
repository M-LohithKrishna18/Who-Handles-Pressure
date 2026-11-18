# Who-Handles-Pressure

Which T20 players deliver when it matters? Built for SupeRR Selector Season 2, Round 1.

A data-driven system that measures how T20 batters and bowlers perform in high-pressure match situations
compared with what an average player would do in exactly the same situation, and ranks the most reliable
performers under pressure (IPL + T20I, 2022–2026).

## Idea in one paragraph
A ball is **high-pressure** when its outcome can swing the match. A win-probability model (logistic regression,
one per innings) scores every ball; **leverage** = the expected swing in win probability across all possible outcomes
of that ball. The top 15% of balls by leverage within each phase × innings are pressure balls (knockouts ×1.5,
super overs automatic). Pressure balls move win probability **2.5×** more than ordinary balls. Players are rated by
**runs above average (RAA)** on pressure balls against out-of-fold LightGBM baselines that know the situation
(phase, wickets, required rate, venue par, season, competition, matchup, build-up pressure such as dot-ball runs)
but not the player. Ratings are shrunk (empirical Bayes) toward each player's own long-run record, with recency
weights. External pressure is read through behaviour: technique drift, loss of control, repeated mistakes.

## Run it
```bash
pip install -r requirements.txt
# make a data/ folder next to the scripts and copy the competition files into it with these names:
#   ball_by_ball.csv matches.csv competitions.csv teams.csv playing_eleven.csv competition_team_player_mapping.csv
./run_all.sh
```
Outputs land in `out/`: auction-available lists `final_bat_list.csv`, `final_bowl_list.csv`; unrestricted lists `unrestricted_bat_list.csv`, `unrestricted_bowl_list.csv`. Runtime: under 5 minutes on 2 cores.

## Pipeline
| Script | What it does |
|---|---|
| `s0_clean.py` | Drop abandoned matches; flag no-results, ties, rain-revised, knockouts, super overs; legal balls; Pakistan/Bangladesh and inactive flags |
| `s0b_exclusions.py` | Availability: IPL 2027 retained players, Indian domestic rule, news-checked retirements and bans |
| `s1_wp.py`, `s1b_tune.py`, `s1c_wp_final.py`, `wp_features.py` | Match state, Elo, venue par; LightGBM vs logistic comparison; final logistic win-probability model |
| `s2_pressure.py` | Leverage per ball, pressure balls, build-up situations, separation checks |
| `s3_expected.py`, `s3b_wicket_value.py` | Out-of-fold expected runs / wicket baselines; wicket value in runs (WP-based in chases) |
| `s4_ratings.py` | RAA per 100 pressure balls, weights, shrinkage toward own record, consistency, drop-off, recent form |
| `s6_select.py` | Candidate pools (available, in form, core-competition evidence, IPL evidence for bowlers) |
| `s7_reliability.py` | Split-half reliability, match bootstrap, sensitivity checks |
| `s8_final.py` | Final score = 0.8 × pressure rating + 0.2 × recent form (same units) |
| `s9_behaviour.py` | Behaviour lens for external pressure (drift, control, repeat-the-mistake) |
| `s10_unrestricted.py` | Unrestricted top 10 with no availability filter (the brief read literally) |

## Honest limits
Pressure skill is only weakly measurable in 2–5 seasons (split-half reliability ≈ 0.2 for batters, ≈ 0 for bowlers);
ranks 4–10 are close calls. Near run-outs, middling and fielding awareness are not in the data. See the methodology report.

## Data
The competition data is not included in this repository. `retained_ipl2027.txt` lists IPL 2027 retentions
(public news). Author: built with Claude (Anthropic) as an analysis assistant.
