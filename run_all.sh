#!/usr/bin/env bash
# Run from the repo folder. Put the six competition CSVs in data/ first (see README).
set -e
cd "$(dirname "$0")"
mkdir -p out
for s in s0_clean s0b_exclusions s1_wp s1b_tune s1c_wp_final s2_pressure s3_expected s3b_wicket_value s4_ratings s6_select s7_reliability s8_final s9_behaviour s10_unrestricted; do
  echo "== $s"; python3 "$s.py"
done
echo "Final lists: out/final_*_list.csv; unrestricted: out/unrestricted_*_list.csv"
