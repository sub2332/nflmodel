#!/usr/bin/env python3
"""
NFL Game Winner / Win Probability Predictor
===========================================
Requires:  pip install nflreadpy pandas numpy scikit-learn pyarrow

Pipeline
--------
 1. Load REG-season schedules, play-by-play, injuries, depth charts (2016-2026, 2020 left out)
 2. Build per-team-per-game stats (ANY/A, EPA/play)
 3. Build strictly PRE-GAME rolling features (EWMA / rolling, shift(1))
 4. Strength of schedule (SOS) + SOS-adjusted net EPA
 5. Common-opponent differential
 6. Elo ratings (reset to 1500 each season, pre-game values only)
 7. Team-specific home-field advantage
 8. Pre-game injury impact (starting-QB flag + weighted key-position score)
 9. Home-minus-Away matchup features -> Logistic Regression
10. Train 2022-24 (with optional group weight multipliers), evaluate 2025-26, print coefficients
11. Predict the next unplayed week of the 2026 season
12. Write a shareable HTML report: game cards with expandable stat comparisons
    and season offense/defense rankings (open in any browser / send to a friend)

Leakage rules: every rolling/cumulative team feature is computed on the team's
chronological game sequence and shifted by one game, so a game's own result is
never used for its own features. Injury reports are published before kickoff,
so they are used as-is for the same week.
"""

import base64
import html
import os
import re
import urllib.request
import webbrowser
import warnings
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DIAG_START_SEASON = 2012                   # earliest season loaded (more history = steadier tests and tuning)
EXCLUDE_SEASONS = [2020]                   # left out on purpose (COVID season)
SEASONS = [s for s in range(DIAG_START_SEASON, 2027) if s not in EXCLUDE_SEASONS]   # 2012-2026 minus 2020
# WALK-FORWARD TESTING: every season from WF_START_SEASON on is predicted by a model trained ONLY on the seasons
# before it, and all those predictions are pooled (about 2,000+ games instead of one small test split).
WF_START_SEASON = 2018
WF_MIN_TRAIN_GAMES = 150
# Elo settings and unit-stat settings are tuned on the seasons BEFORE the walk-forward period only, so the
# walk-forward results never use information from the games they are scored on.
TUNE_SEASONS = [s for s in SEASONS if s < WF_START_SEASON]
TRAIN_SEASONS = TUNE_SEASONS                     # (kept for older code paths)
TEST_SEASONS = [s for s in SEASONS if s >= WF_START_SEASON]
LIVE_MODEL_USES_ALL_GAMES = True                 # the model that predicts the upcoming week is fit on every completed game
PREDICT_SEASON = 2026
PREDICT_WEEK = None            # which week to predict. None = automatic (the earliest week that still has an unplayed game)
                               # "next" = the week AFTER the latest week that has been played (e.g. Monday night of week 11 -> week 12)
                               # or a number such as 13. Injury reports and depth charts for a later week appear midweek,
                               # so until then that week's picks assume everyone is healthy.
_w = os.environ.get("NFL_PREDICT_WEEK", "").strip()      # automatic runs (GitHub) can set the week without editing this file
if _w:
    PREDICT_WEEK = "next" if _w.lower() == "next" else int(_w)

EWMA_SPAN = 6                  # ANY/A exponentially weighted span
EPA_WINDOW = 6                 # rolling EPA window (games)
COMMON_OPP_PRIOR_SEASON_BEFORE_WEEK = 5   # use prior season too if week < 5

LEAGUE_ANYA = 5.8              # neutral prior when a team has no history yet
LEAGUE_EPA = 0.0

# ---- Predicted-score settings ----
# ---- Extra test factors (each is its own factor group; the factor-accuracy tables show whether it helps) ----
# Weather: the schedule has the temperature / wind that were actually recorded at each game. Upcoming games have no
# recorded weather yet, so enter a forecast here to use it live, e.g. {"BUF@KC": {"temp": 22, "wind": 18}}
# (key = "AWAY@HOME"). Games with no weather are treated as calm and mild. Domes / closed roofs ignore weather.
WEATHER_OVERRIDES = {}
WEATHER_FORECAST = True        # fetch a forecast (Open-Meteo, free, no key) for each outdoor game being predicted, show it on the
                               # game card, and use it in the points total. Falls back to "calm and mild" if it cannot be reached.
                               # An entry in WEATHER_OVERRIDES wins over the fetched forecast.
WIND_FADE_WEEKS = 8            # roster-continuity effects fade out over the first this-many weeks of a season
LEAGUE_GO_RATE = 0.30          # league rate of going for it on 4th-and-short near midfield (prior for each team)
GO_PRIOR_TRIALS = 12
LEAGUE_PLAYS = 62.0            # neutral prior for offensive plays per game
DIVISIONS = {
    "BUF": "AE", "MIA": "AE", "NE": "AE", "NYJ": "AE", "BAL": "AN", "CIN": "AN", "CLE": "AN", "PIT": "AN",
    "HOU": "AS", "IND": "AS", "JAX": "AS", "TEN": "AS", "DEN": "AW", "KC": "AW", "LV": "AW", "LAC": "AW",
    "DAL": "NE", "NYG": "NE", "PHI": "NE", "WAS": "NE", "CHI": "NN", "DET": "NN", "GB": "NN", "MIN": "NN",
    "ATL": "NS", "CAR": "NS", "NO": "NS", "TB": "NS", "ARI": "NW", "LAR": "NW", "SF": "NW", "SEA": "NW"}
MARKET_FLAG_PTS = 8.0           # flag a game when the model's win probability differs from the Vegas moneyline by this many points
# ---- three new tests (each can be switched off on its own to compare) ----
GARBAGE_TIME_FILTER = True     # ignore 4th-quarter plays once one side has under GARBAGE_WP win probability (blowout time)
GARBAGE_WP = 0.05
EARLY_DOWN_SUCCESS = True      # add early-down (1st/2nd down) success rate to the unit ratings (takes half of success rate's weight)
INJURY_PLAYER_IMPACT = True    # injuries weighted by WHO is out (position value x how much he actually plays), as ONE number per team
                               # False = the old position-group counts (6 separate inputs)
EPA_ELO = False                # (tested: no gain, redundant with Elo) add a second Elo that learns from EPA margin (blended with the score margin) instead of the score alone
EPA_ELO_BLEND = 0.5            # 0 = same as the normal Elo, 1 = learn only from EPA margin
MARGIN_SD = 13.5               # std-dev of NFL game margins; converts win probability to a point margin
TOTAL_RIDGE_ALPHA = 50.0       # regularization for the projected-total-points model

# ---- Turnover / explosive-play settings ----
TOV_WINDOW = 8                 # turnover margin per game, averaged over the last N games
EXPL_RUN_YDS = 10              # explosive run  = 10+ yards
EXPL_PASS_YDS = 20             # explosive pass = 20+ yards (completed, not a sack)
LEAGUE_EXPL = 0.09             # neutral prior: share of scrimmage plays that are explosive
LEAGUE_SUCC = 0.45             # neutral prior: share of plays with positive EPA (success rate)

# ---- Offense / defense unit rankings (these carry REAL model weight) ----
# Each unit's league rank blends the metrics that best reveal how good an offense or defense
# really is. EPA/play, success rate and ANY/A are the core efficiency measures; the others
# cover what they miss: the run game, pass protection / pass rush, finishing drives, 3rd downs.
#   epa      EPA/play
#   succ     success rate (share of plays with positive EPA)
#   anya     ANY/A (passing efficiency incl. sacks, TDs, interceptions)
#   rushepa  rush EPA/play (run game / run defense, which ANY/A ignores)
#   press    pressure rate (sacks + QB hits per dropback): protection allowed / pass rush generated
#   drive    scoring-drive rate (share of drives ending in a TD or FG)
#   expl     explosive-play rate
#   third    3rd-down conversion rate
#   rz       red-zone TD % (offense: TDs scored on trips; defense: TD rate allowed)
#   ppg      points per game scored / allowed
#   ypg      scrimmage yards per game gained / allowed (net of sacks)
#   tov      turnovers: offense = giveaways per game (fewer is better), defense = takeaways per game
# These stats enter the model ONLY through this rating (there are no separate EPA / ANY/A /
# explosive features), so each one is counted exactly once.
# Set a weight to 0 to drop a stat from the rating. Weights are normalized to sum to 1.
# Points and yards are included at modest weight (they reflect pace and game script as well as quality).
# Turnovers (giveaways / takeaways) are part of the rating too, so there is no separate turnover group.
# Weights below follow the factor-accuracy test on 2021-26 data: the stats that predicted winners best
# (success rate, EPA, points, yards, ANY/A, scoring drives) get the most; pressure rate, red zone,
# turnovers and 3rd-down rate were the weaker stats in testing, so they are kept small.
RANK_STAT_WEIGHTS = {"epa": 0.16, "succ": 0.08, "esucc": 0.08, "anya": 0.12, "ppg": 0.12, "ypg": 0.08, "drive": 0.10,
                     "rushepa": 0.06, "expl": 0.06, "press": 0.04, "rz": 0.03, "tov": 0.03, "third": 0.04}
# (is HIGHER better for the offense stat?, is HIGHER better for the defense stat?)
if not EARLY_DOWN_SUCCESS:        # fold early-down success back into overall success rate
    RANK_STAT_WEIGHTS["succ"] += RANK_STAT_WEIGHTS.pop("esucc")
RANK_DIRECTION = {"epa": (True, False), "succ": (True, False), "esucc": (True, False), "anya": (True, False),
                  "rushepa": (True, False), "press": (False, True), "drive": (True, False),
                  "expl": (True, False), "third": (True, False), "rz": (True, False),
                  "ppg": (True, False), "ypg": (True, False), "tov": (False, True),
                  "prate": (True, False)}
LEAGUE_UNIT_PRIORS = {"rushepa": -0.05, "press": 0.28, "drive": 0.38, "third": 0.40,
                      "ppg": 22.0, "ypg": 330.0, "tov": 1.2, "prate": 90.0}  # Week-1 neutral values
# display: key -> (offense label, defense label, number format, display scale)
RANK_STAT_META = {
    "epa": ("EPA/play", "EPA allowed", "{:+.3f}", 1),
    "succ": ("Success %", "Success % allowed", "{:.1f}", 100),
    "esucc": ("Early-down success %", "Early-down success % allowed", "{:.1f}", 100),
    "anya": ("ANY/A", "ANY/A allowed", "{:.2f}", 1),
    "rushepa": ("Rush EPA", "Rush EPA allowed", "{:+.3f}", 1),
    "press": ("Pressure allowed %", "Pressure generated %", "{:.1f}", 100),
    "drive": ("Scoring drive %", "Scoring drive % allowed", "{:.1f}", 100),
    "expl": ("Explosive %", "Explosive % allowed", "{:.1f}", 100),
    "third": ("3rd down %", "3rd down % allowed", "{:.1f}", 100),
    "rz": ("Red-zone TD %", "Red-zone TD % allowed", "{:.1f}", 100),
    "ppg": ("Points / game", "Points allowed / game", "{:.1f}", 1),
    "ypg": ("Yards / game", "Yards allowed / game", "{:.0f}", 1),
    "tov": ("Giveaways / game", "Takeaways / game", "{:.2f}", 1),
    "prate": ("Passer rating", "Passer rating allowed", "{:.1f}", 1),
}
RANK_STAT_NAMES = {"epa": "EPA/play", "succ": "success rate", "esucc": "early-down success rate", "anya": "ANY/A", "rushepa": "rush EPA/play",
                   "press": "pressure rate", "drive": "scoring-drive rate",
                   "expl": "explosive-play rate", "third": "3rd-down rate", "rz": "red-zone TD %",
                   "ppg": "points per game", "ypg": "yards per game",
                   "tov": "turnovers (giveaways / takeaways)", "prate": "passer rating"}

# Which games feed the unit ratings (offense / defense rankings):
#   "season_minus_g1" = every game of the CURRENT season before this one, except the team's first game of
#                       the season (a team's 2nd game uses game 1 only because nothing else exists yet; a
#                       team's 1st game of a season falls back to the last-6-games average, as it has no
#                       current-season data)
#   "recent"          = the last ~6 games (rolls across seasons)
# Only the unit ratings use this; every other feature keeps its own window.
UNIT_STATS_MODE = "season_minus_g1"
UNIT_WINDOW_TEXT = ""          # filled in by set_unit_config()

# ---- Unit-stat refinements (all switchable) ----
UNIT_AUTO_TUNE = True          # True = TEST the settings below on the training seasons and use the best;
                               # False = use UNIT_STATS_MODE / EARLY_PRIOR_GAMES / OPP_ADJUST exactly as written.
EARLY_PRIOR_GAMES = 4          # early-season blend: last season's average counts like this many games
                               # (0 = off). It fades out as the new season's games pile up.
OPP_ADJUST = 0.5               # opponent adjustment strength: 0 = off, 1 = fully remove opponent quality
UNIT_HALFLIFE = 5                # used by mode "ewma": a game this many games old counts half as much
UNIT_HALFLIFE_CANDIDATES = [3, 5, 8]
UNIT_MODE_CANDIDATES = ["recent", "season_minus_g1", "ewma"]
UNIT_PRIOR_CANDIDATES = [0, 4, 8]
UNIT_ADJ_CANDIDATES = [0.0, 0.5, 1.0]


def set_unit_config(mode, prior, adj, halflife=None):
    """Set the unit-stat settings and the plain-English description the report shows."""
    global UNIT_STATS_MODE, EARLY_PRIOR_GAMES, OPP_ADJUST, UNIT_WINDOW_TEXT, UNIT_HALFLIFE
    UNIT_STATS_MODE, EARLY_PRIOR_GAMES, OPP_ADJUST = mode, prior, adj
    if halflife is not None:
        UNIT_HALFLIFE = halflife
    if mode == "season_minus_g1":
        t = "every game this season except the team's first"
    elif mode == "ewma":
        t = f"all recent games, recent ones counting more (a game {UNIT_HALFLIFE:g} games old counts half)"
    else:
        t = "each team's recent games (about the last 6)"
    if mode == "season_minus_g1" and prior > 0:
        t += f", blended early in the year with last season's averages (counted as {prior:g} games)"
    if adj > 0:
        t += f", adjusted for opponent strength ({adj:.0%})"
    UNIT_WINDOW_TEXT = t


set_unit_config(UNIT_STATS_MODE, EARLY_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE)

# ---- Red-zone TD% settings (pre-game rolling rate, shrunk toward league average) ----
LEAGUE_RZ_TD = 0.57            # league-average red-zone TD rate used as the prior
RZ_WINDOW = 8                  # last N games
RZ_PRIOR_TRIPS = 12            # prior weight, in 'trips' (red-zone trips are few, so rates are noisy)

# ---- Elo settings ----
ELO_START = 1500.0             # every team starts each season here
ELO_K = 20.0                   # max points swing per game (before any MOV multiplier)
ELO_HOME_ADV = 48.0            # Elo points added to the home team's rating for win-prob
ELO_SEASON_CARRYOVER = None    # share of last season's edge over/under 1500 a team keeps each new season.
                               # None = TEST several values on the training seasons and use the best.
                               # 0.0 = hard reset to 1500 every season; e.g. 0.67 keeps two-thirds.
ELO_CARRYOVER_CANDIDATES = [0.0, 0.25, 0.5, 0.67, 0.8]
ELO_USE_MOV = False            # True = scale updates by margin of victory (538-style)
ELO_AUTO_TUNE = True           # True = also TEST K / home-field points / margin-of-victory on the training
                               # seasons (with the carryover) and use the best; False = use the values above.
ELO_EARLY_BOOST = 1.0          # multiply K by this in the first ELO_EARLY_WEEKS of a season (ratings are least
ELO_EARLY_WEEKS = 4            # certain early, so they can move faster). 1.0 = off.
ELO_K_CANDIDATES = [10.0, 15.0, 20.0, 25.0, 30.0]
ELO_HFA_CANDIDATES = [30.0, 48.0, 65.0]
ELO_MOV_CANDIDATES = [False, True]
ELO_EARLY_BOOST_CANDIDATES = [1.0, 1.25, 1.5, 2.0]
ELO_QB_SCALE = 0.0             # QB-aware Elo: Elo points per 1.0 EPA/dropback that today's QB is above/below the team's
                               # usual QB (a backup => negative). 0 = off. Tuned automatically when ELO_AUTO_TUNE is on.
ELO_QB_SCALE_CANDIDATES = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0]

# ---- Rest, travel and team-specific home field, inside Elo (all in Elo points; 25 Elo is roughly 1 point on the spread) ----
# Each is tuned automatically on the training seasons when ELO_AUTO_TUNE is on (candidates below; [0.0] = off).
ELO_BYE_PTS = 0.0              # bonus for a team coming off a bye or long rest (10+ days since its last game)
ELO_SHORT_PTS = 0.0            # penalty for a short week (5 or fewer days since its last game, e.g. Thursday games)
ELO_TRAVEL_PTS = 0.0           # extra home edge per 1,000 miles the visiting team travelled
ELO_HFA_LR = 0.0               # dynamic team home-field: how fast a team's own home edge moves with its home results
ELO_HFA_CAP = 40.0             # a team's own home edge can't move more than this from the league figure
ELO_HFA_RETAIN = 0.7           # share of a team's own home edge kept into the next season
# ---- Offense / defense points ratings (a second, different view of team strength) ----
PTS_RATINGS_ENABLED = True     # each team gets an offense rating and a defense rating (points above/below average),
                               # updated after every game; projects both teams' scores. Tuned on the tuning seasons.
PTS_LR_CANDIDATES = [0.03, 0.05, 0.08, 0.12, 0.18]      # how fast the ratings move
PTS_CARRY_CANDIDATES = [0.3, 0.5, 0.7, 0.85]            # share kept into the next season
PTS_HFA_CANDIDATES = [1.0, 2.0, 3.0]                    # home-field edge, in points
ELO_BYE_CANDIDATES = [0.0, 10.0, 20.0, 30.0]
ELO_SHORT_CANDIDATES = [0.0, 10.0, 20.0, 30.0]
ELO_TRAVEL_CANDIDATES = [0.0, 5.0, 10.0, 20.0, 30.0]
ELO_HFA_LR_CANDIDATES = [0.0, 1.0, 2.0, 4.0, 6.0]

# approximate home-stadium location (latitude, longitude) for the travel-distance estimate
TEAM_COORDS = {
    "ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79),
    "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52), "CLE": (41.51, -81.70),
    "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06),
    "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64), "KC": (39.05, -94.48),
    "LV": (36.09, -115.18), "LAC": (33.95, -118.34), "LAR": (33.95, -118.34), "MIA": (25.96, -80.24),
    "MIN": (44.97, -93.26), "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07),
    "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SF": (37.40, -121.97),
    "SEA": (47.60, -122.33), "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86)}

# ---- Vegas spread as a model input ----
USE_VEGAS_FEATURE = False       # True = the closing spread is one more model factor (the model is then partly
                               # market-based). False = the model stays independent of Vegas.

# ---- Home-field advantage / report settings ----
REPORT_FILENAME = "nfl_predictions.html"   # written next to this script
OPEN_REPORT_IN_BROWSER = not os.environ.get("CI")          # never try to open a browser on an automatic (GitHub) run
LOGO_MODE = "embed"            # "embed": download each team logo once and bake it into the HTML
                               #          (works offline and in anything you send; ~1 MB file)
                               # "link" : hotlink the logo URLs (small file, viewer needs internet)
                               # "none" : text only
LOGO_PX = 96                   # logo download size in pixels (embed mode)
LEAGUE_HFA_PTS = 2.0           # league-average home-field edge (points) used as the prior
HFA_SHRINK_GAMES = 34          # team HFA is shrunk toward the league value (~2 seasons of games)

# Injury weights (see build_injury_table)
QB_WEIGHT = 4.0                # starting QB out, when the backup is a replacement-level downgrade
#   The real QB penalty = QB_WEIGHT x (how much worse the backup is than the starter):
#   gap = (starter quality - backup quality) / QB_DROP_SCALE, clipped to [QB_DROP_MIN, QB_DROP_MAX].
#   Quality = EPA per dropback from PRIOR games, shrunk toward QB_PRIOR_EPA (so unknown or
#   barely-played backups count as replacement level). A good backup => small penalty.
QB_PRIOR_EPA = -0.05           # assumed quality of an unknown QB (EPA/dropback)
QB_PRIOR_DROPBACKS = 100       # how many dropbacks of evidence it takes to outweigh that prior
QB_DROP_SCALE = 0.25           # a 0.25 EPA/dropback downgrade = full QB_WEIGHT
QB_DROP_MIN, QB_DROP_MAX = 0.25, 1.5
KEY_SKILL_WEIGHT = 2.0         # WR1 / RB1
WR2_WEIGHT = 1.0               # WR2
WR3_WEIGHT = 0.25              # WR3 counts for less
WR_DEPTH_WEIGHT = 0.1          # any other WR
RB_BACKUP_SNAP_MIN = 0.30      # a non-RB1 back only counts if he averaged this share of offensive snaps
RB_BACKUP_WEIGHT = 1.0         #   (over his last 6 games); below that share he counts for 0
RB_NO_SNAP_DATA_WEIGHT = 0.5   # used for non-RB1 backs only if snap counts could not be loaded
DEPTH_WEIGHT = 0.5             # unknown positions
STATUS_FACTOR = {"Out": 1.0, "Doubtful": 0.75}
# Injuries by position group: each group gets its OWN learned weight in the model instead of one hand-set total.
INJURY_STAR_QUALITY = True     # pass rushers / linebackers who are out count more if they are productive (pressures per game, PFR)
INJURY_WR_QUALITY = True       # missing WR / TE count more when they are heavily targeted (targets per game, last 12 games)
INJURY_CB_QUALITY = True       # missing corners count more or less by how well they have covered (yards allowed per target, PFR)
INJURY_CARRY_LAST_WEEK = False # TEST ONLY: when the week being predicted has no injury reports yet, reuse the latest week's report
                               # as if it were this week's (to see how picks move). Do not leave on for real picks.
INJ_CARRIED_FROM = 0           # set by the code when the above is in use
# Players you KNOW are out that the data does not have yet. {week: {team: [(player name, status), ...]}}.
# Only used when that week is the one being predicted, so it cannot leak into a later week. Status: "Out" or "Doubtful".
# Team abbreviations as on the report cards (LA = Rams, GB = Packers, KC = Chiefs ...).
MANUAL_INJURIES = {
    # 12: {"LA": [("Davante Adams", "Out"), ("Puka Nacua", "Out")]},     <- example; remove the # to use
}
INJURY_RECEIVING_SHARE = True  # its own factor: the share of a team's targets that belongs to receivers who are out
                               # (so losing a 40%-of-targets star with nobody behind him hurts a lot)
INJURY_BY_POSITION = not INJURY_PLAYER_IMPACT
KEY_PLAYER_W = 1.2             # a missing player at or above this weight is called out by name on the game cards
INJ_WEEK_NOW = 0               # the week those two tables belong to
INJ_KEY_NOW = {}               # {team: [(weight, name, position, status)]} for the week being predicted (filled by build_injury_table)
INJ_IMPACT_NOW = {}            # {(team, gsis_id): weight} same week, used by the injury report tables
INJ_GROUPS = {"ol": ["T", "OT", "LT", "RT", "G", "OG", "C", "OL"],
              "dl": ["DE", "OLB", "EDGE", "DT", "NT", "DL"],
              "lb": ["LB", "ILB", "MLB"],
              "db": ["CB", "S", "FS", "SS", "DB"],
              "rec": ["WR", "TE"],
              "rb": ["RB", "FB", "HB"]}
INJ_GROUP_COLS = [f"inj_{g}" for g in INJ_GROUPS]
INJ_GROUP_COLS_DIFF = [c + "_diff" for c in INJ_GROUP_COLS]
POS_WEIGHTS = {
    # tackles (injury reports say T/OT, not LT/RT)
    "T": 2.0, "OT": 2.0, "LT": 2.0, "RT": 2.0,
    # edge rushers (reports say DE / OLB, so those are used as the EDGE proxy)
    "DE": 2.0, "OLB": 2.0, "EDGE": 2.0,
    "CB": 2.0,
    "DT": 1.0, "NT": 1.0, "DL": 1.0,
    "LB": 1.0, "ILB": 1.0, "MLB": 1.0,
    "S": 1.0, "FS": 1.0, "SS": 1.0, "DB": 1.0,
    "G": 1.0, "OG": 1.0, "C": 1.0, "OL": 1.0,
    "TE": 1.0,
    "K": 0.1, "P": 0.1, "LS": 0.1,
}

# sos_adj_net_epa_diff == epa_off_diff - epa_def_diff + sos_diff (exact linear
# combination). Including all of them in one model makes coefficients
# unidentifiable, so by default the model uses sos_diff and the SOS-adjusted
# net EPA is shown in the output table. Flip to True to swap sos_diff out.
USE_SOS_ADJ_IN_MODEL = False

# False = follow the spec literally (fit on 2022-24 only).
# True  = refit on every completed game before predicting upcoming week.
REFIT_ON_ALL_COMPLETED = False

# ---- Manual weight tuning: ONE multiplier per factor, all listed here ----
# Each factor's LEARNED weight is multiplied by the number next to it:
#   1.0 = leave as learned, above 1 = lean on it more, below 1 = lean on it less, 0 = ignore it.
# After scaling, the model is re-calibrated so probabilities stay sensible.
# All offense/defense stats (EPA/play, success rate, ANY/A, rush EPA, pressure, scoring drives,
# explosive plays, 3rd down, red zone, points, yards, turnovers) are stacked inside "Unit rankings", so each is counted exactly once.
# Their relative mix is set in RANK_STAT_WEIGHTS near the top of the file.
# Which factor groups the model actually uses. None = all of them (the full blend).
# A group left out is not fit at all (its multiplier below is ignored).
# Current setup: Elo + Unit rankings + QB quality + Injuries (all at 1.0 = learned weights). To go back to the full model set this to None.
ACTIVE_FEATURE_GROUPS = ["Elo", "Points ratings", "Unit rankings", "QB quality", "Injuries"] + (["EPA Elo"] if EPA_ELO else []) + (["Receiving injuries"] if INJURY_RECEIVING_SHARE else [])
# Tested, no help ("Style matchups" is shown on the game cards as notes instead): "Weather", "Coaching", "Roster continuity", "Divisional", "Style matchups"
   # add "QB passer rating" to retest it

# Full-blend weights before the simple test (to restore): Unit rankings 1.6, SOS 0.5, Common Opponents 0.5,
# Elo 1.2, QB quality 1.0, Injuries 0.9, Home field 1.0, Team HFA 0.
MODEL_C = 0.05    # logistic regularization strength of the live / walk-forward model (smaller = steadier; tested best)
GROUP_WEIGHT_MULTIPLIERS = {
    "Unit rankings": 1.0,        # offense vs defense quality incl. turnovers (mix set in RANK_STAT_WEIGHTS)
    "SOS": 0.0,                  # strength of schedule (added almost nothing in testing, so kept small)
    "Common Opponents": 0.0,    # margin / EPA against shared opponents (mostly redundant with Elo + units)
    "Elo": 1.0,                  # Elo rating
    "EPA Elo": 1.0,              # Elo that learns from EPA margin (blended with score margin)
    "Weather": 1.0,              # wind x QB-quality gap (the clean wind / cold effect on totals is in the total model)
    "Coaching": 1.0,             # 4th-down aggressiveness gap
    "Roster continuity": 1.0,    # returning offensive production and new-QB flag, fading after the first weeks
    "Style matchups": 1.0,       # pass-vs-pass, rush-vs-rush and pass-rush-vs-protection edges
    "Divisional": 1.0,           # divisional games are closer than the ratings say
    "Points ratings": 1.0,       # offense / defense points ratings: projected margin before any home edge
    "QB passer rating": 1.0,     # TEST weight: starting QB's passer rating, home minus away (backup's if starter out)
    "QB quality": 1.0,          # expected starter's EPA/dropback (backup's if the starter is out)
    "Receiving injuries": 1.0,   # raise (e.g. 2.0) to make missing receivers matter more
    "Injuries": (1.0 if INJURY_PLAYER_IMPACT else 0.5),   # starting QB out + weighted key-player injuries
    "Home field": 0.5,           # plain home indicator
    "Team HFA": 0.0,             # team-specific home-field edge (hurt accuracy in testing, so switched off)
    "Vegas line": 1.0,           # closing point spread (only used when USE_VEGAS_FEATURE is True)
}

TEAM_FIX = {"STL": "LAR", "LA": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS",
            "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX"}

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "wp", "qtr",
            "play_type", "epa", "qb_dropback", "pass_attempt", "sack",
            "yards_gained", "pass_touchdown", "interception",
            "two_point_attempt", "qb_scramble", "passer_player_id",
            "rusher_player_id", "receiver_player_id",
            "fumble_lost", "third_down_converted", "third_down_failed",
            "drive", "yardline_100", "touchdown", "td_team", "play_id",
            "qb_hit", "fixed_drive", "fixed_drive_result", "complete_pass", "down", "ydstogo"]


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------
def to_pandas(df):
    """nflreadpy returns polars; convert to pandas."""
    return df if isinstance(df, pd.DataFrame) else df.to_pandas()


def subset_columns(df, cols):
    """Keep only existing columns (works for polars or pandas) to save memory."""
    keep = [c for c in cols if c in df.columns]
    return df[keep] if isinstance(df, pd.DataFrame) else df.select(keep)


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fix_team(series):
    return series.replace(TEAM_FIX)


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def american_to_prob(o):
    """American moneyline odds -> implied win probability (includes the bookmaker's margin)."""
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def vegas_home_prob(df):
    """No-vig home win probability from the two moneylines (NaN when they are missing)."""
    if "home_moneyline" not in df.columns or "away_moneyline" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ph, pa = american_to_prob(df["home_moneyline"]), american_to_prob(df["away_moneyline"])
    return pd.Series(ph / (ph + pa), index=df.index)


def load_schedules():
    sched = to_pandas(nfl.load_schedules(SEASONS))
    sched = sched[sched["game_type"] == "REG"].copy()
    sched["home_team"] = fix_team(sched["home_team"])
    sched["away_team"] = fix_team(sched["away_team"])
    if "result" not in sched.columns:
        sched["result"] = sched["home_score"] - sched["away_score"]
    if "location" not in sched.columns:
        sched["location"] = "Home"
    # sanity-check the spread's sign against the moneylines (positive spread should mean home favored)
    if "spread_line" in sched.columns and "home_moneyline" in sched.columns:
        ph = vegas_home_prob(sched)
        ok = ph.notna() & sched["spread_line"].notna() & (sched["spread_line"] != 0)
        if ok.sum() > 50 and (((sched["spread_line"] > 0) == (ph > 0.5))[ok]).mean() < 0.3:
            print("  ! spread_line has the opposite sign to what was assumed; flipping it")
            sched["spread_line"] = -sched["spread_line"]
    return sched.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def load_pbp(game_ids):
    frames = []
    for s in SEASONS:                       # season-by-season so one failure doesn't kill the run
        try:
            raw = nfl.load_pbp([s])
        except Exception as e:              # e.g. season not published yet
            print(f"  ! play-by-play for {s} unavailable ({e}); skipping")
            continue
        frames.append(to_pandas(subset_columns(raw, PBP_COLS)))
    pbp = pd.concat(frames, ignore_index=True)
    for c in PBP_COLS:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pbp = pbp[pbp["game_id"].isin(game_ids)].copy()
    pbp["posteam"] = fix_team(pbp["posteam"])
    pbp["defteam"] = fix_team(pbp["defteam"])
    pbp["td_team"] = fix_team(pbp["td_team"])
    if pbp["play_id"].notna().any():          # keep plays in game order (drive starts rely on it)
        pbp = pbp.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    return pbp


def small_logo_url(url):
    """ESPN's image combiner serves a resized copy, much lighter than the 500px original."""
    m = re.match(r"https?://a\.espncdn\.com(/i/teamlogos/.+)", url or "")
    return f"https://a.espncdn.com/combiner/i?img={m.group(1)}&h={LOGO_PX}&w={LOGO_PX}" if m else url


def download_data_uri(url, timeout=6):
    """Fetch an image and return it as a base64 data URI (None on any failure)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data, ctype = resp.read(), resp.headers.get_content_type()
        if data and ctype.startswith("image/"):
            return f"data:{ctype};base64," + base64.b64encode(data).decode()
    except Exception:
        pass
    return None


def load_team_logos():
    """team abbreviation -> image source (data URI when embedded, otherwise a URL)."""
    if LOGO_MODE == "none":
        return {}
    try:
        teams = to_pandas(nfl.load_teams())
    except Exception as exc:
        print(f"  ! load_teams failed ({exc}); logos skipped")
        return {}
    url_col = first_col(teams, ["team_logo_espn", "team_logo_squared", "team_logo_wikipedia"])
    if url_col is None or "team_abbr" not in teams.columns:
        print("  ! team logo columns not found; logos skipped")
        return {}
    teams = teams.dropna(subset=[url_col]).copy()
    teams["team_abbr"] = fix_team(teams["team_abbr"])
    urls = dict(zip(teams["team_abbr"], teams[url_col]))
    if LOGO_MODE != "embed":
        return urls

    out, fails = {}, 0
    for abbr, url in urls.items():
        data = None
        if fails < 3:                                   # offline? stop retrying after 3 straight failures
            small = small_logo_url(url)
            data = download_data_uri(small) or (download_data_uri(url) if small != url else None)
            fails = 0 if data else fails + 1
        out[abbr] = data or url                         # fall back to hotlinking
    n = sum(v.startswith("data:") for v in out.values())
    print(f"  Logos: {n} of {len(out)} embedded" + ("" if n == len(out) else " (the rest link online)"))
    return out


def load_position_map():
    """gsis_id -> position (used to find true WR1 / RB1)."""
    try:
        pl = to_pandas(nfl.load_players())
        pos_col = first_col(pl, ["position", "pos"])
        return dict(zip(pl["gsis_id"], pl[pos_col].astype(str).str.upper())) if pos_col else {}
    except Exception as e:
        print(f"  ! load_players failed ({e}); WR1/RB1 will not be position-filtered")
        return {}


def load_depth_qb1():
    """(team, season, week) -> QB1 gsis_id from depth charts (fallback only)."""
    try:
        dc = to_pandas(nfl.load_depth_charts(SEASONS))
    except Exception as e:
        print(f"  ! load_depth_charts failed ({e}); using play-by-play starters only")
        return {}
    team_c = first_col(dc, ["club_code", "team", "team_abbr"])
    pos_c = first_col(dc, ["depth_position", "position", "pos_abb"])
    rank_c = first_col(dc, ["depth_team", "pos_rank"])
    id_c = first_col(dc, ["gsis_id", "player_id"])
    if not all([team_c, pos_c, rank_c, id_c]) or "week" not in dc.columns:
        return {}
    qb = dc[(dc[pos_c].astype(str).str.upper() == "QB")
            & (dc[rank_c].astype(str).str.strip().isin(["1", "1.0"]))].copy()
    qb[team_c] = fix_team(qb[team_c])
    qb = qb.drop_duplicates([team_c, "season", "week"])
    return {(t, s, w): i for t, s, w, i in zip(qb[team_c], qb["season"], qb["week"], qb[id_c])}


def load_depth_current_qbs():
    """
    Latest depth-chart snapshot -> {team: (QB1 gsis_id, QB2 gsis_id), ...}, plus the snapshot time.
    nflreadpy's depth charts are timestamped snapshots with NO season/week columns, so this is
    used ONLY for the upcoming week (it would leak the future if applied to past games).
    Columns: dt, team, gsis_id, pos_abb, pos_rank.
    """
    try:
        dc = to_pandas(nfl.load_depth_charts(SEASONS))
    except Exception as e:
        print(f"  ! load_depth_charts failed ({e}); using play-by-play for starter / backup QB")
        return {}, None
    need = ["dt", "team", "gsis_id", "pos_abb", "pos_rank"]
    if any(c not in dc.columns for c in need):
        print(f"  ! depth chart columns not as expected ({list(dc.columns)}); using play-by-play for QBs")
        return {}, None
    dc = dc[dc["pos_abb"].astype(str).str.upper() == "QB"].copy()
    dc["dt"] = pd.to_datetime(dc["dt"], utc=True, errors="coerce")
    dc["rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    dc = dc.dropna(subset=["dt", "rank", "gsis_id"])
    if dc.empty:
        return {}, None
    dc["team"] = fix_team(dc["team"])
    dc = dc[dc["dt"] == dc.groupby("team")["dt"].transform("max")]          # latest snapshot per team
    out = {}
    for t, d in dc.sort_values("rank").groupby("team"):
        ids = list(dict.fromkeys(d["gsis_id"]))                              # QB1, QB2 in order
        out[t] = (ids[0], ids[1] if len(ids) > 1 else None)
    return out, dc["dt"].max()


def pick_week(df):
    """The week to predict, following PREDICT_WEEK (df needs season, week, result). None if nothing is left to play."""
    s_ = df[df["season"] == PREDICT_SEASON]
    up = s_[s_["result"].isna()]
    if up.empty:
        return None
    auto = int(up["week"].min())
    if PREDICT_WEEK is None:
        return auto
    if PREDICT_WEEK == "next":
        played = s_[s_["result"].notna()]
        nxt = int(played["week"].max()) + 1 if len(played) else auto
        return nxt if (s_["week"] == nxt).any() else auto
    w = int(PREDICT_WEEK)
    if (s_["week"] == w).any():
        return w
    print(f"  ! PREDICT_WEEK = {PREDICT_WEEK} has no games in the schedule; using week {auto}")
    return auto


def apply_depth_chart_qbs(tg, sched, depth, snap_time):
    """Override qb_id / qb2_id for the UPCOMING week only (handles trades and new starters)."""
    if not depth or snap_time is None:
        return tg
    up = sched[(sched["season"] == PREDICT_SEASON) & sched["result"].isna()]
    if up.empty:
        return tg
    wk = pick_week(sched)
    first_day = pd.to_datetime(up[up["week"] == wk]["gameday"], errors="coerce").min()
    if pd.notna(first_day) and snap_time < first_day.tz_localize("UTC") - pd.Timedelta(days=21):
        print(f"  ! depth chart snapshot ({snap_time:%Y-%m-%d}) is too old for week {wk}; ignoring it")
        return tg
    m = (tg["season"] == PREDICT_SEASON) & (tg["week"] == wk) & tg["pts_for"].isna()
    n = 0
    for i in tg.index[m]:
        q = depth.get(tg.at[i, "team"])
        if q:
            tg.at[i, "qb_id"] = q[0] or tg.at[i, "qb_id"]
            tg.at[i, "qb2_id"] = q[1] if q[1] else tg.at[i, "qb2_id"]
            n += 1
    print(f"  Depth chart ({snap_time:%Y-%m-%d}) set starter/backup QB for {n} teams in week {wk}")
    return tg


# ----------------------------------------------------------------------------
# 2. PER-GAME TEAM STATS FROM PLAY-BY-PLAY
# ----------------------------------------------------------------------------
def prepare_dropbacks(pbp):
    """Dropback plays with ANY/A numerator (num) and denominator (den) columns."""
    pb = pbp[(pbp["play_type"] == "pass") & (pbp["qb_dropback"] == 1)
             & (pbp["two_point_attempt"].fillna(0) != 1)
             & pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    yds = pb["yards_gained"].fillna(0)
    sack = pb["sack"].fillna(0) == 1
    att = (pb["pass_attempt"].fillna(0) == 1) & ~sack
    pb["att"] = att.astype(int)
    pb["sk"] = sack.astype(int)
    pb["num"] = (np.where(att, yds, 0)
                 + 20 * ((pb["pass_touchdown"].fillna(0) == 1) & att).astype(int)
                 - 45 * (pb["interception"].fillna(0) == 1).astype(int)
                 - np.where(sack, -yds, 0))
    pb["den"] = pb["att"] + pb["sk"]
    return pb


def red_zone_drives(pbp):
    """
    One row per red-zone TRIP: a (game, drive, offense) with at least one run/pass
    snap from the opponent's 20 or closer. `td` = 1 if that offense scored a
    touchdown on the drive (defensive/return TDs do not count).
    """
    rz = pbp[pbp["play_type"].isin(["pass", "run"]) & (pbp["yardline_100"] <= 20)
             & pbp["posteam"].notna() & pbp["defteam"].notna() & pbp["drive"].notna()
             & (pbp["two_point_attempt"].fillna(0) != 1)].copy()
    rz["td"] = ((rz["touchdown"].fillna(0) == 1) & (rz["td_team"] == rz["posteam"])).astype(int)
    return rz.groupby(["game_id", "drive", "posteam", "defteam"], as_index=False)["td"].max()


def drive_starts(pbp):
    """
    Starting field position of each drive: yardline_100 (yards from the opponent's
    end zone) of the drive's first scrimmage play. Own-yard-line = 100 - yardline_100.
    """
    d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna() & pbp["defteam"].notna()
            & pbp["drive"].notna() & pbp["yardline_100"].notna()]
    return d.groupby(["game_id", "drive", "posteam", "defteam"], as_index=False)["yardline_100"].first()


UNIT_GAME_COLS = ["off_prate", "def_prate", "off_rush_epa", "def_rush_epa", "off_press", "def_press",
                  "off_drive", "def_drive", "off_third", "def_third", "off_yds", "def_yds"]


def scoring_drives(pbp):
    """
    One row per drive with score = 1 when the offense ended it with a TD or field goal.
    Returns None when the play-by-play has no drive-result data.
    """
    if pbp["fixed_drive_result"].notna().sum() == 0:
        return None
    dcol = "fixed_drive" if pbp["fixed_drive"].notna().any() else "drive"
    d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna() & pbp["defteam"].notna()
            & pbp[dcol].notna() & pbp["fixed_drive_result"].notna()]
    g = d.groupby(["game_id", dcol, "posteam", "defteam"], as_index=False)["fixed_drive_result"].first()
    g["score"] = g["fixed_drive_result"].isin(["Touchdown", "Field goal"]).astype(int)
    return g


def build_game_stats(pbp):
    """
    One row per (game_id, team) with:
      off_anya / def_anya : ANY/A produced / allowed
      off_epa  / def_epa  : EPA per play produced / allowed (offensive snaps)

    ANY/A = (PassYds + 20*PassTD - 45*INT - SackYds) / (Attempts + Sacks)
    Sacks are handled explicitly: attempts exclude sack plays (whether or not
    the dataset flags sacks as pass_attempt), the denominator adds sacks back,
    and sack yards lost = -yards_gained on sack == 1 plays.
    """
    # ---- garbage time: drop 4th-quarter plays once the game is (almost) decided ----
    if GARBAGE_TIME_FILTER and pbp["wp"].notna().any():
        q_ = pd.to_numeric(pbp["qtr"], errors="coerce")
        w_ = pd.to_numeric(pbp["wp"], errors="coerce")
        gt = (q_ >= 4) & ((w_ < GARBAGE_WP) | (w_ > 1 - GARBAGE_WP))
        print(f"  Garbage-time filter: ignoring {int(gt.sum()):,} late blowout plays ({gt.mean():.1%} of all plays).")
        pbp = pbp[~gt.fillna(False)]

    # ---- ANY/A ----
    pb = prepare_dropbacks(pbp)

    def anya(group_col, name):
        g = pb.groupby(["game_id", group_col])[["num", "den"]].sum().reset_index()
        g[name] = g["num"] / g["den"].where(g["den"] > 0)
        return g.rename(columns={group_col: "team"})[["game_id", "team", name]]

    # ---- EPA/play ----
    ep = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()
             & pbp["posteam"].notna() & pbp["defteam"].notna()]

    def epa(group_col, name):
        g = ep.groupby(["game_id", group_col])["epa"].mean().reset_index()
        return g.rename(columns={group_col: "team", "epa": name})

    trips = red_zone_drives(pbp)

    def rz(group_col, prefix):
        g = trips.groupby(["game_id", group_col]).agg(
            **{f"{prefix}_rz_trips": ("td", "size"), f"{prefix}_rz_td": ("td", "sum")}).reset_index()
        return g.rename(columns={group_col: "team"})

    # ---- turnovers (INT + fumbles lost on run/pass plays) and explosive plays ----
    sc = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna() & pbp["defteam"].notna()
             & (pbp["two_point_attempt"].fillna(0) != 1)].copy()
    y = sc["yards_gained"].fillna(0)
    is_pass = sc["play_type"] == "pass"
    sc["tov"] = (sc["interception"].fillna(0) == 1).astype(int) + (sc["fumble_lost"].fillna(0) == 1).astype(int)
    sc["expl"] = (((~is_pass) & (y >= EXPL_RUN_YDS))
                  | (is_pass & (sc["sack"].fillna(0) != 1) & (y >= EXPL_PASS_YDS))).astype(int)
    sc["one"] = 1

    def sc_stats(group_col, tov_name, expl_name):
        g = sc.groupby(["game_id", group_col]).agg(
            **{tov_name: ("tov", "sum"), "e": ("expl", "sum"), "n": ("one", "sum")}).reset_index()
        g[expl_name] = g["e"] / g["n"]                         # explosive plays per scrimmage play
        return g.rename(columns={group_col: "team"})[["game_id", "team", tov_name, expl_name]]

    def succ(group_col, name):                       # success rate = share of plays with EPA > 0
        g = ep.assign(s=(ep["epa"] > 0).astype(int)).groupby(["game_id", group_col])["s"].mean().reset_index()
        return g.rename(columns={group_col: "team", "s": name})

    def esucc(group_col, name):                      # early-down (1st / 2nd down) success rate
        e_ = ep[ep["down"].isin([1, 2])]
        g = e_.assign(s=(e_["epa"] > 0).astype(int)).groupby(["game_id", group_col])["s"].mean().reset_index()
        return g.rename(columns={group_col: "team", "s": name})

    def epa_total(name):                             # total offensive EPA in the game (for the EPA-based Elo)
        g = ep.groupby(["game_id", "posteam"])["epa"].sum().reset_index()
        return g.rename(columns={"posteam": "team", "epa": name})

    parts = [anya("posteam", "off_anya"), anya("defteam", "def_anya"),
             epa("posteam", "off_epa"), epa("defteam", "def_epa"),
             rz("posteam", "off"), rz("defteam", "def"),
             sc_stats("posteam", "tov_given", "off_expl_rate"),       # giveaways; explosive rate made
             sc_stats("defteam", "tov_taken", "def_expl_rate"),       # takeaways; explosive rate allowed
             succ("posteam", "off_succ"), succ("defteam", "def_succ"),   # success rate made / allowed
             esucc("posteam", "off_esucc"), esucc("defteam", "def_esucc"), epa_total("off_epa_tot")]
    # ---- extra unit metrics: rush EPA, pressure rate, scoring-drive rate, 3rd-down rate ----
    rush = ep[ep["play_type"] == "run"]

    def rush_epa(group_col, name):
        g = rush.groupby(["game_id", group_col])["epa"].mean().reset_index()
        return g.rename(columns={group_col: "team", "epa": name})

    passes = ep[ep["play_type"] == "pass"]

    def pass_epa(group_col, name):
        g = passes.groupby(["game_id", group_col])["epa"].mean().reset_index()
        return g.rename(columns={group_col: "team", "epa": name})

    pr = pb.assign(pr=((pb["sk"] == 1) | (pb["qb_hit"].fillna(0) == 1)).astype(int))

    def pressure(group_col, name):                   # offense: pressure allowed; defense: pressure generated
        g = pr.groupby(["game_id", group_col])["pr"].mean().reset_index()
        return g.rename(columns={group_col: "team", "pr": name})

    sd = scoring_drives(pbp)

    def drive_rate(group_col, name):
        g = sd.groupby(["game_id", group_col])["score"].mean().reset_index()
        return g.rename(columns={group_col: "team", "score": name})

    conv = sc["third_down_converted"].fillna(0) == 1
    t3 = sc[["game_id", "posteam", "defteam"]].assign(c=conv.astype(int),
                                                      a=(conv | (sc["third_down_failed"].fillna(0) == 1)).astype(int))

    def third(group_col, name):
        g = t3.groupby(["game_id", group_col])[["c", "a"]].sum().reset_index()
        g[name] = g["c"] / g["a"].where(g["a"] > 0)
        return g.rename(columns={group_col: "team"})[["game_id", "team", name]]

    def yards(group_col, name):                      # scrimmage yards (run + pass, sack yards subtracted)
        g = sc.groupby(["game_id", group_col])["yards_gained"].sum().reset_index()
        return g.rename(columns={group_col: "team", "yards_gained": name})

    def passer_rating(group_col, name):              # NFL passer rating from attempts, completions, yards, TD, INT
        a = pb[pb["att"] == 1]
        g = a.assign(c=a["complete_pass"].fillna(0), y=a["yards_gained"].fillna(0),
                     t=(a["pass_touchdown"].fillna(0) == 1).astype(int),
                     i=(a["interception"].fillna(0) == 1).astype(int), n=1
                     ).groupby(["game_id", group_col])[["c", "y", "t", "i", "n"]].sum().reset_index()
        n = g["n"].where(g["n"] > 0)
        r = lambda x: x.clip(0, 2.375)
        g[name] = (r((g["c"] / n - 0.3) * 5) + r((g["y"] / n - 3) * 0.25) + r(g["t"] / n * 20)
                   + r(2.375 - g["i"] / n * 25)) / 6 * 100
        if a["complete_pass"].notna().sum() == 0:
            g[name] = np.nan                         # no completion data in the play-by-play
        return g.rename(columns={group_col: "team"})[["game_id", "team", name]]

    extra = [passer_rating("posteam", "off_prate"), passer_rating("defteam", "def_prate"),
             yards("posteam", "off_yds"), yards("defteam", "def_yds"),
             rush_epa("posteam", "off_rush_epa"), rush_epa("defteam", "def_rush_epa"),
             pass_epa("posteam", "off_pass_epa"), pass_epa("defteam", "def_pass_epa"),
             pressure("posteam", "off_press"), pressure("defteam", "def_press"),
             third("posteam", "off_third"), third("defteam", "def_third")]
    if sd is not None:
        extra += [drive_rate("posteam", "off_drive"), drive_rate("defteam", "def_drive")]

    # ---- coaching / scheme: pace (offensive plays) and 4th-down aggressiveness near midfield ----
    plays = sc.groupby(["game_id", "posteam"]).size().rename("off_plays").reset_index().rename(columns={"posteam": "team"})
    f4 = pbp[(pbp["down"] == 4) & (pbp["ydstogo"] <= 3) & pbp["yardline_100"].between(35, 65)
             & pbp["posteam"].notna() & pbp["play_type"].isin(["pass", "run", "punt", "field_goal"])]
    g4 = (f4.assign(go=f4["play_type"].isin(["pass", "run"]).astype(int)).groupby(["game_id", "posteam"])
            .agg(g4_go=("go", "sum"), g4_n=("go", "size")).reset_index().rename(columns={"posteam": "team"}))
    extra += [plays, g4]

    out = parts[0]
    for p in parts[1:] + extra:
        out = out.merge(p, on=["game_id", "team"], how="outer")
    for c in ["off_plays", "g4_go", "g4_n", "off_pass_epa", "def_pass_epa", "off_esucc", "def_esucc", "off_epa_tot"]:
        if c not in out.columns:
            out[c] = np.nan
    for c in UNIT_GAME_COLS:
        if c not in out.columns:
            out[c] = np.nan
    unavailable = [c for c in UNIT_GAME_COLS if out[c].isna().all()]
    if unavailable:
        print(f"  ! no play-by-play data for {unavailable}; those stats count as league average in the unit ratings")
    return out


def build_team_games(sched, game_stats):
    """Long format: one row per team per REG game (played or scheduled)."""
    def side(team, opp, is_home, pf, pa):
        return pd.DataFrame({"game_id": sched["game_id"], "season": sched["season"],
                             "week": sched["week"], "team": sched[team], "opp": sched[opp],
                             "is_home": is_home, "pts_for": sched[pf], "pts_against": sched[pa]})
    tg = pd.concat([side("home_team", "away_team", 1, "home_score", "away_score"),
                    side("away_team", "home_team", 0, "away_score", "home_score")])
    tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    tg["gnum"] = tg.groupby("team").cumcount()       # chronological game index per team
    return tg.merge(game_stats, on=["game_id", "team"], how="left")


# ----------------------------------------------------------------------------
# 3 + 4. PRE-GAME ROLLING FEATURES, SOS, SOS-ADJUSTED NET EPA
# ----------------------------------------------------------------------------
def add_pregame_features(tg):
    g = tg.groupby("team", sort=False)

    def ewma(s):   # EWMA of games BEFORE this one
        return s.ewm(span=EWMA_SPAN, adjust=False, ignore_na=True).mean().shift(1)

    def roll(s):   # rolling mean of the previous EPA_WINDOW games
        return s.shift(1).rolling(EPA_WINDOW, min_periods=1).mean()

    tg["anya_off_pre"] = g["off_anya"].transform(ewma)
    tg["anya_def_pre"] = g["def_anya"].transform(ewma)       # ANY/A allowed (lower = better)
    tg["epa_off_pre"] = g["off_epa"].transform(roll)
    tg["epa_def_pre"] = g["def_epa"].transform(roll)         # EPA allowed (lower = better)

    # neutral priors for teams with no history (2022 Week 1)
    tg[["anya_off_pre", "anya_def_pre"]] = tg[["anya_off_pre", "anya_def_pre"]].fillna(LEAGUE_ANYA)
    tg[["epa_off_pre", "epa_def_pre"]] = tg[["epa_off_pre", "epa_def_pre"]].fillna(LEAGUE_EPA)
    tg["net_epa_pre"] = tg["epa_off_pre"] - tg["epa_def_pre"]

    # ---- style-matchup inputs: pass / rush EPA and pressure, split by unit (last 8 prior games) ----
    for k, c in (("pass_off", "off_pass_epa"), ("pass_def", "def_pass_epa"), ("rush_off", "off_rush_epa"),
                 ("rush_def", "def_rush_epa"), ("press_off", "off_press"), ("press_def", "def_press")):
        v = g[c].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
        tg["sty_" + k + "_pre"] = v.fillna(v.mean() if v.notna().any() else 0.0)

    # ---- red-zone TD%: rolling TDs / trips over the last RZ_WINDOW games (prior games only),
    #      shrunk toward the league rate because a team has few trips per game ----
    def rz_rate(td_col, trips_col):
        td = g[td_col].transform(lambda s: s.fillna(0).shift(1).rolling(RZ_WINDOW, min_periods=1).sum()).fillna(0)
        tr = g[trips_col].transform(lambda s: s.fillna(0).shift(1).rolling(RZ_WINDOW, min_periods=1).sum()).fillna(0)
        return (td + LEAGUE_RZ_TD * RZ_PRIOR_TRIPS) / (tr + RZ_PRIOR_TRIPS)
    tg["rz_off_pre"] = rz_rate("off_rz_td", "off_rz_trips")     # offense: TD rate scoring
    tg["rz_def_pre"] = rz_rate("def_rz_td", "def_rz_trips")     # defense: TD rate ALLOWED (lower = better)

    # ---- explosive plays (rolling rate, prior games only) and turnover margin per game ----
    tg["expl_off_pre"] = g["off_expl_rate"].transform(roll).fillna(LEAGUE_EXPL)
    tg["expl_def_pre"] = g["def_expl_rate"].transform(roll).fillna(LEAGUE_EXPL)   # allowed (lower = better)
    tg["succ_off_pre"] = g["off_succ"].transform(roll).fillna(LEAGUE_SUCC)
    tg["succ_def_pre"] = g["def_succ"].transform(roll).fillna(LEAGUE_SUCC)        # allowed (lower = better)
    tg["esucc_off_pre"] = g["off_esucc"].transform(roll).fillna(LEAGUE_SUCC)
    tg["esucc_def_pre"] = g["def_esucc"].transform(roll).fillna(LEAGUE_SUCC)
    for k, col_o, col_d in (("rushepa", "off_rush_epa", "def_rush_epa"), ("press", "off_press", "def_press"),
                            ("drive", "off_drive", "def_drive"), ("third", "off_third", "def_third"),
                            ("ppg", "pts_for", "pts_against"), ("ypg", "off_yds", "def_yds"),
                            ("tov", "tov_given", "tov_taken"), ("prate", "off_prate", "def_prate")):
        tg[f"{k}_off_pre"] = g[col_o].transform(roll).fillna(LEAGUE_UNIT_PRIORS[k])
        tg[f"{k}_def_pre"] = g[col_d].transform(roll).fillna(LEAGUE_UNIT_PRIORS[k])
    tg["plays_pre"] = g["off_plays"].transform(roll).fillna(LEAGUE_PLAYS)
    _go = g["g4_go"].transform(lambda x: x.fillna(0).shift(1).rolling(24, min_periods=1).sum()).fillna(0)
    _gn = g["g4_n"].transform(lambda x: x.fillna(0).shift(1).rolling(24, min_periods=1).sum()).fillna(0)
    tg["go_pre"] = (_go + LEAGUE_GO_RATE * GO_PRIOR_TRIALS) / (_gn + GO_PRIOR_TRIALS)      # 4th-down go rate (shrunk)
    tg["tov_margin_game"] = tg["tov_taken"] - tg["tov_given"]
    tg["tov_margin_pre"] = (tg.groupby("team", sort=False)["tov_margin_game"]
                              .transform(lambda s: s.shift(1).rolling(TOV_WINDOW, min_periods=1).mean())
                              .fillna(0.0))

    # ---- SOS: mean pre-game net EPA of opponents already faced this season ----
    opp = tg[["game_id", "team", "net_epa_pre"]].rename(
        columns={"team": "opp", "net_epa_pre": "opp_net_epa_pre"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")
    tg["sos"] = (tg.groupby(["team", "season"])["opp_net_epa_pre"]
                   .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
                   .fillna(0.0))
    tg["sos_adj_net_epa"] = tg["net_epa_pre"] + tg["sos"]
    return tg


UNIT_SOURCES = {   # unit stat -> (offense per-game column, defense per-game column)
    "epa": ("off_epa", "def_epa"), "succ": ("off_succ", "def_succ"), "esucc": ("off_esucc", "def_esucc"),
    "anya": ("off_anya", "def_anya"),
    "rushepa": ("off_rush_epa", "def_rush_epa"), "press": ("off_press", "def_press"),
    "drive": ("off_drive", "def_drive"), "third": ("off_third", "def_third"),
    "expl": ("off_expl_rate", "def_expl_rate"), "ppg": ("pts_for", "pts_against"),
    "ypg": ("off_yds", "def_yds"), "tov": ("tov_given", "tov_taken"),
    "prate": ("off_prate", "def_prate"),
}


def _season_pre(v, gi, grp_keys):
    """Prior-games-only average within each season, skipping the team's first game (falls back to that
    first game alone for the second game). Returns (mean or NaN, number of games used)."""
    v = v.astype(float)
    ok = v.notna().astype(float)
    one = v.fillna(0.0)

    def cs(x):
        return x.groupby(grp_keys).cumsum().groupby(grp_keys).shift(1)
    later = (gi >= 1)
    ms, mn = cs(one.where(later, 0.0)), cs(ok.where(later, 0.0))
    fs, fn = cs(one), cs(ok)
    use_main = mn > 0
    n = np.where(use_main, mn, fn)
    tot = np.where(use_main, ms, fs)
    n = np.nan_to_num(n)
    mean = pd.Series(np.where(n > 0, tot / np.where(n > 0, n, 1), np.nan), index=v.index)
    return mean, pd.Series(n, index=v.index)


def add_unit_inputs(tg, mode=None, prior_games=None, adj=None, halflife=None):
    """
    Build the ur_<stat>_off_pre / ur_<stat>_def_pre columns the unit ratings rank.
      mode "recent"         : last-6-games values.
      mode "season_minus_g1": average of this season's earlier games EXCLUDING the team's first game.
      prior_games (>0)      : blend LAST season's average in, worth that many games (fades as games pile up).
      adj (>0)              : opponent adjustment. Each game's number is corrected by how much the opponent
                              normally pushes that number (their pre-game average vs the league's that week).
    Everything uses prior games only, so there is no leakage.
    """
    mode = UNIT_STATS_MODE if mode is None else mode
    prior_games = EARLY_PRIOR_GAMES if prior_games is None else prior_games
    adj = OPP_ADJUST if adj is None else adj
    hl = UNIT_HALFLIFE if halflife is None else halflife
    tg = tg.copy()
    keys = list(RANK_STAT_WEIGHTS)
    gi = tg.groupby(["team", "season"]).cumcount()            # 0 = team's first game this season
    grp_keys = [tg["team"], tg["season"]]
    key = pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]])
    opp_key = pd.MultiIndex.from_arrays([tg["game_id"], tg["opp"]])
    wk = [tg["season"], tg["week"]]

    def pre_from(v, fallback):
        """pre-game value of a per-game series under the chosen mode (+ optional last-season blend)."""
        if mode == "ewma":
            r = v.groupby(tg["team"]).transform(
                lambda x: x.shift(1).ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
            return r.fillna(fallback)
        if mode != "season_minus_g1":
            r = v.groupby(tg["team"]).transform(lambda x: x.shift(1).rolling(EPA_WINDOW, min_periods=1).mean())
            return r.fillna(fallback)
        mean, n = _season_pre(v, gi, grp_keys)
        if prior_games > 0:
            tsm = v.groupby([tg["team"], tg["season"]]).mean()
            pr = tsm.groupby(level=0).shift(1).reindex(pd.MultiIndex.from_arrays([tg["team"], tg["season"]])).values
            pr = pd.Series(pr, index=v.index)
            blended = (n * mean.fillna(0.0) + prior_games * pr) / (n + prior_games)
            mean = mean.where(pr.isna(), blended)
        return mean.fillna(fallback)

    for k in keys:
        if k == "rz":
            if mode != "season_minus_g1":
                for side in ("off", "def"):
                    tg[f"ur_rz_{side}_pre"] = tg[f"rz_{side}_pre"]
                continue
            for side, td, tr in (("off", "off_rz_td", "off_rz_trips"), ("def", "def_rz_td", "def_rz_trips")):
                def cum(col):
                    v = tg[col].fillna(0.0).astype(float)
                    sk = v.where(gi >= 1, 0.0)
                    main = sk.groupby(grp_keys).cumsum().groupby(grp_keys).shift(1)
                    first = v.groupby(grp_keys).cumsum().groupby(grp_keys).shift(1)
                    return main.where(main > 0, first)
                t_, n_ = cum(td), cum(tr)
                rate = (t_.fillna(0) + LEAGUE_RZ_TD * RZ_PRIOR_TRIPS) / (n_.fillna(0) + RZ_PRIOR_TRIPS)
                tg[f"ur_rz_{side}_pre"] = rate.where(n_.notna(), tg[f"rz_{side}_pre"])
            continue
        co, cd = UNIT_SOURCES[k]
        raw_o, raw_d = tg[co].astype(float), tg[cd].astype(float)
        fb_o, fb_d = tg[f"{k}_off_pre"], tg[f"{k}_def_pre"]
        if mode == "recent" and adj <= 0:                    # exactly the old "recent" columns
            tg[f"ur_{k}_off_pre"], tg[f"ur_{k}_def_pre"] = fb_o, fb_d
            continue
        po, pd_ = pre_from(raw_o, fb_o), pre_from(raw_d, fb_d)
        if adj > 0:
            def opp_of(x):
                return pd.Series(x.values, index=key).reindex(opp_key).values
            d_off = pd.Series(opp_of(pd_), index=tg.index) - pd_.groupby(wk).transform("mean")   # opp defense vs league
            d_def = pd.Series(opp_of(po), index=tg.index) - po.groupby(wk).transform("mean")     # opp offense vs league
            adj_o = raw_o - adj * d_off.fillna(0.0)
            adj_d = raw_d - adj * d_def.fillna(0.0)
            po, pd_ = pre_from(adj_o, fb_o), pre_from(adj_d, fb_d)
        tg[f"ur_{k}_off_pre"], tg[f"ur_{k}_def_pre"] = po, pd_
    return tg


def _unit_cv_score(tg):
    """Season-by-season (leave-one-season-out) log-loss on the TRAINING seasons of a 2-feature model that uses
    only the unit-rating matchups, for whatever unit inputs tg currently holds."""
    t, _ = add_unit_ranks(tg)
    t = t[["game_id", "season", "is_home", "off_unit", "def_unit", "pts_for", "pts_against"]]
    h = t[t["is_home"] == 1].set_index("game_id")
    a = t[t["is_home"] == 0].set_index("game_id")
    d = pd.DataFrame({"season": h["season"], "x1": h["off_unit"] - a["def_unit"],
                      "x2": a["off_unit"] - h["def_unit"], "m": h["pts_for"] - h["pts_against"]}).dropna()
    d = d[(d["m"] != 0) & d["season"].isin(TUNE_SEASONS)]
    y = (d["m"] > 0).astype(int).values
    X = d[["x1", "x2"]].values
    seas = d["season"].values
    loss, hit = [], []
    for sn in np.unique(seas):
        te = seas == sn
        m = LogisticRegression(C=1.0, max_iter=500).fit(X[~te], y[~te])
        p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        loss.append(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        hit.append((p >= 0.5) == y[te])
    return float(np.concatenate(loss).mean()), float(np.concatenate(hit).mean()), len(y)


def choose_unit_config(tg):
    """Try the unit-stat settings (window / recency weighting, last-season blend, opponent adjustment) on the
    TUNING seasons only and apply the best one. Prints the whole table so you can see what each piece is worth."""
    if not UNIT_AUTO_TUNE:
        set_unit_config(UNIT_STATS_MODE, EARLY_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE)
        print(f"\nUnit-stat settings (fixed): {UNIT_WINDOW_TEXT}")
        return
    print("\n" + "=" * 78)
    print(f"UNIT-STAT SETTINGS TEST (offense/defense ratings alone, tuning seasons {TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]}, "
          "season by season)")
    print("=" * 78)
    print(f"  {'window':<16s} {'half-life':>9s} {'last-season blend':>17s} {'opp. adjust':>11s} {'log-loss':>9s} {'accuracy':>9s}")
    configs = []
    for mode in UNIT_MODE_CANDIDATES:
        for hl in (UNIT_HALFLIFE_CANDIDATES if mode == "ewma" else [None]):
            for pg in (UNIT_PRIOR_CANDIDATES if mode == "season_minus_g1" else [0]):
                for ad in UNIT_ADJ_CANDIDATES:
                    configs.append((mode, pg, ad, hl))
    res = []
    for mode, pg, ad, hl in configs:
        try:
            ll, acc, n = _unit_cv_score(add_unit_inputs(tg, mode, pg, ad, hl))
        except Exception as ex:
            print(f"  ! {mode}/{pg}/{ad}/{hl} skipped ({ex})")
            continue
        res.append((ll, acc, mode, pg, ad, hl))
        print(f"  {mode:<16s} {('-' if hl is None else f'{hl:g}'):>9s} {pg:>17g} {ad:>11.1f} {ll:9.4f} {acc:9.3f}")
    if not res:
        set_unit_config(UNIT_STATS_MODE, EARLY_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE)
        return
    ll, acc, mode, pg, ad, hl = min(res, key=lambda r: r[0])
    base = [r for r in res if r[2] == "season_minus_g1" and r[3] == 0 and r[4] == 0.0]
    if base:
        print(f"  Plain 'this season minus game 1' (no blend, no adjustment): log-loss {base[0][0]:.4f}")
    set_unit_config(mode, pg, ad, hl)
    print(f"  Using: {UNIT_WINDOW_TEXT}  (log-loss {ll:.4f}).")
    print("  To go back, set UNIT_AUTO_TUNE = False and EARLY_PRIOR_GAMES = 0, OPP_ADJUST = 0, UNIT_STATS_MODE = 'season_minus_g1'.")


# ----------------------------------------------------------------------------
# 4b. OFFENSE / DEFENSE UNIT RANKINGS (pre-game, league-relative)
# ----------------------------------------------------------------------------
OFF_UNIT_COLS = [f"ur_{k}_off_pre" for k in RANK_STAT_WEIGHTS]
DEF_UNIT_COLS = [f"ur_{k}_def_pre" for k in RANK_STAT_WEIGHTS]


def add_unit_ranks(tg):
    """
    Rank every team's offense and defense against the whole league, each week, using
    only pre-game numbers (so no leakage).

      off_unit / def_unit : 0-1 rating = weighted average of the team's league percentile on
                            each key stat in RANK_STAT_WEIGHTS (1 = best in the league)
      off_rank / def_rank : 1-32 league rank of that rating (1 = best)

    A complete team x week grid is used so teams on a bye are ranked too; their value
    at a bye is the one they carry into their next game.
    Returns (tg with the four new columns, the full grid for display).
    """
    bad = set(RANK_STAT_WEIGHTS) - set(RANK_DIRECTION)
    if bad:
        raise ValueError(f"Unknown RANK_STAT_WEIGHTS keys {sorted(bad)}; choose from {sorted(RANK_DIRECTION)}")
    cols = OFF_UNIT_COLS + DEF_UNIT_COLS
    weeks = tg[["season", "week"]].drop_duplicates()
    full = tg[["team", "season"]].drop_duplicates().merge(weeks, on="season")
    full = full.merge(tg[["team", "season", "week"] + cols], on=["team", "season", "week"], how="left")
    full = full.sort_values(["team", "season", "week"]).reset_index(drop=True)
    full[cols] = full.groupby(["team", "season"])[cols].bfill()
    full[cols] = full.groupby(["team", "season"])[cols].ffill()

    tot = sum(RANK_STAT_WEIGHTS.values())
    grp = full.groupby(["season", "week"])
    # percentile rank where 1.0 = best in the league; each stat's direction comes from RANK_DIRECTION
    off = sum(w / tot * grp[f"ur_{k}_off_pre"].rank(pct=True, ascending=RANK_DIRECTION[k][0])
              for k, w in RANK_STAT_WEIGHTS.items())
    dfn = sum(w / tot * grp[f"ur_{k}_def_pre"].rank(pct=True, ascending=RANK_DIRECTION[k][1])
              for k, w in RANK_STAT_WEIGHTS.items())
    full["off_unit"], full["def_unit"] = off.fillna(0.5), dfn.fillna(0.5)
    grp = full.groupby(["season", "week"])
    full["off_rank"] = grp["off_unit"].rank(ascending=False, method="min")
    full["def_rank"] = grp["def_unit"].rank(ascending=False, method="min")

    keep = ["team", "season", "week", "off_unit", "def_unit", "off_rank", "def_rank"]
    return tg.merge(full[keep], on=["team", "season", "week"], how="left"), full


# ----------------------------------------------------------------------------
# 5. COMMON OPPONENTS
# ----------------------------------------------------------------------------
def compute_common_opponents(matchups, tg):
    played = tg[tg["pts_for"].notna()].copy()
    played["margin"] = played["pts_for"] - played["pts_against"]
    played["game_net_epa"] = played["off_epa"] - played["def_epa"]
    cols = ["season", "week", "opp", "margin", "game_net_epa"]
    by_team = {t: d[cols] for t, d in played.groupby("team")}
    empty = pd.DataFrame(columns=cols)

    def prior(team, season, week):
        d = by_team.get(team)
        if d is None:
            return empty
        mask = (d["season"] == season) & (d["week"] < week)
        if week < COMMON_OPP_PRIOR_SEASON_BEFORE_WEEK:
            mask = mask | (d["season"] == season - 1)
        return d[mask]

    rows = []
    for gid, s, w, ht, at in zip(matchups["game_id"], matchups["season"], matchups["week"],
                                 matchups["home_team"], matchups["away_team"]):
        h, a = prior(ht, s, w), prior(at, s, w)
        common = set(h["opp"]) & set(a["opp"])
        if common:
            hc, ac = h[h["opp"].isin(common)], a[a["opp"].isin(common)]
            hm, am = hc["margin"].mean(), ac["margin"].mean()
            he, ae = hc["game_net_epa"].mean(), ac["game_net_epa"].mean()
            m, e = hm - am, he - ae
            rows.append((gid, 0.0 if pd.isna(m) else m, 0.0 if pd.isna(e) else e, 1, len(common),
                         hm, am, he, ae))
        else:
            rows.append((gid, 0.0, 0.0, 0, 0, np.nan, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["game_id", "common_opp_margin_diff",
                                       "common_opp_epa_diff", "has_common_opponents",
                                       "n_common_opps", "h_common_margin", "a_common_margin",
                                       "h_common_epa", "a_common_epa"])


# ----------------------------------------------------------------------------
# 5b. ELO RATINGS
# ----------------------------------------------------------------------------
def _miles_between(a, b):
    """Great-circle distance in miles between two TEAM_COORDS entries (0 if either is unknown)."""
    if a not in TEAM_COORDS or b not in TEAM_COORDS:
        return 0.0
    la1, lo1, la2, lo2 = map(np.radians, (*TEAM_COORDS[a], *TEAM_COORDS[b]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(3959 * 2 * np.arcsin(np.sqrt(h)))


def build_elo_context(sched):
    """
    Per game (indexed by game_id): days of rest each team had since its previous game (7 = normal or a new
    season) and the miles the visiting team travels (0 at neutral sites). Uses only the schedule dates.
    """
    g = sched[["game_id", "season", "gameday", "home_team", "away_team", "location"]].copy()
    g["day"] = pd.to_datetime(g["gameday"], errors="coerce")
    lg = pd.concat([g.assign(team=g["home_team"], side="h"), g.assign(team=g["away_team"], side="a")])
    lg = lg.sort_values(["team", "day", "game_id"])
    prev = lg.groupby("team")["day"].shift(1)
    same = lg.groupby("team")["season"].shift(1) == lg["season"]
    rest = (lg["day"] - prev).dt.days.where(same)
    lg["rest"] = rest.where(rest.between(1, 30)).fillna(7.0)
    h = lg[lg["side"] == "h"].set_index("game_id")["rest"].rename("rest_h")
    a = lg[lg["side"] == "a"].set_index("game_id")["rest"].rename("rest_a")
    ctx = pd.concat([h, a], axis=1)
    neutral = g.set_index("game_id")["location"].astype(str).str.lower() == "neutral"
    miles = pd.Series([0.0 if neutral[gid] else _miles_between(at, ht)
                       for gid, ht, at in zip(g["game_id"], g["home_team"], g["away_team"])],
                      index=g["game_id"], name="miles")
    return ctx.join(miles).fillna({"rest_h": 7.0, "rest_a": 7.0, "miles": 0.0})


def compute_elo(sched, carryover=None, k=None, hfa=None, mov=None, early=None, qb=None, qb_scale=None,
                ctx=None, bye=None, short=None, travel=None, hfa_lr=None):
    """
    Walk through every REG game in chronological order.

    For each game we first RECORD both teams' pre-game ratings (this is what
    the model sees, so there is no leakage), then update the ratings if the
    game has been played:

        expected_home = 1 / (1 + 10 ** (-(home_elo + HFA - away_elo) / 400))
        delta         = K * (actual - expected_home)        [actual: 1 / 0.5 / 0]
        home_elo     += delta ;  away_elo -= delta

    Pre-game adjustments (in Elo points, all from information known before kickoff):
      QB      : today's QB vs the team's usual QB
      rest    : bye / long rest bonus, short-week penalty
      travel  : more home edge the farther the visitor travels
      team HFA: each team's own home edge, learned from its home results (dynamic home-field advantage)

    Returns (per-game DataFrame, dict of current ratings).
    """
    carry = carryover if carryover is not None else (0.0 if ELO_SEASON_CARRYOVER is None else ELO_SEASON_CARRYOVER)
    k_ = ELO_K if k is None else k
    hfa_pts = ELO_HOME_ADV if hfa is None else hfa
    use_mov = ELO_USE_MOV if mov is None else mov
    early_boost = ELO_EARLY_BOOST if early is None else early
    qs = ELO_QB_SCALE if qb_scale is None else qb_scale
    bye_pts = ELO_BYE_PTS if bye is None else bye
    short_pts = ELO_SHORT_PTS if short is None else short
    trav_pts = ELO_TRAVEL_PTS if travel is None else travel
    lr = ELO_HFA_LR if hfa_lr is None else hfa_lr
    dev_h = dev_a = None
    if qb is not None and qs:                          # per-game QB deviation (EPA/dropback vs the team's usual QB)
        dev_h, dev_a = qb["h_dev"].to_dict(), qb["a_dev"].to_dict()
    rest_h = rest_a = miles = None
    if bye_pts or short_pts or trav_pts:
        if ctx is None:
            ctx = build_elo_context(sched)
        rest_h, rest_a, miles = ctx["rest_h"].to_dict(), ctx["rest_a"].to_dict(), ctx["miles"].to_dict()

    def rest_adj(r):
        return bye_pts if r >= 10 else (-short_pts if r <= 5 else 0.0)

    games = sched.sort_values(["season", "week", "gameday", "game_id"])
    ratings, extra, rows, cur_season = {}, {}, [], None

    for gid, season, week, ht, at, hs, as_, loc in zip(
            games["game_id"], games["season"], games["week"], games["home_team"], games["away_team"],
            games["home_score"], games["away_score"], games["location"]):

        # new season: reset (or partially regress) every team toward 1500
        if season != cur_season:
            if cur_season is not None:
                for t in ratings:
                    ratings[t] = ELO_START + carry * (ratings[t] - ELO_START)
                for t in extra:
                    extra[t] *= ELO_HFA_RETAIN
            cur_season = season

        rh, ra = ratings.get(ht, ELO_START), ratings.get(at, ELO_START)
        home_site = str(loc).lower() != "neutral"
        hfa = hfa_pts if home_site else 0.0
        adj_h = qs * dev_h.get(gid, 0.0) if dev_h is not None else 0.0     # QB adjustment, in Elo points
        adj_a = qs * dev_a.get(gid, 0.0) if dev_a is not None else 0.0
        if rest_h is not None:
            adj_h += rest_adj(rest_h.get(gid, 7.0))
            adj_a += rest_adj(rest_a.get(gid, 7.0))
        home_edge = 0.0                                  # extra home edge: team's own + travel
        if home_site:
            home_edge = extra.get(ht, 0.0) + (trav_pts * min(miles.get(gid, 0.0), 2800.0) / 1000.0
                                              if miles is not None else 0.0)
        elo_gap = (rh + adj_h + hfa + home_edge) - (ra + adj_a)   # home minus away, incl. HFA and adjustments
        p_home = 1.0 / (1.0 + 10 ** (-elo_gap / 400.0))
        rows.append((gid, rh + adj_h + home_edge, ra + adj_a, p_home))     # pre-game values (HFA itself excluded)

        if pd.notna(hs) and pd.notna(as_):              # only update on completed games
            actual = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            mult = 1.0
            if use_mov and hs != as_:
                winner_gap = elo_gap if hs > as_ else -elo_gap
                mult = np.log(abs(hs - as_) + 1) * 2.2 / (winner_gap * 0.001 + 2.2)
            delta = k_ * (early_boost if week <= ELO_EARLY_WEEKS else 1.0) * mult * (actual - p_home)
            ratings[ht], ratings[at] = rh + delta, ra - delta
            if lr and home_site:                         # dynamic team home-field edge
                extra[ht] = float(np.clip(extra.get(ht, 0.0) + lr * (actual - p_home), -ELO_HFA_CAP, ELO_HFA_CAP))

    out = pd.DataFrame(rows, columns=["game_id", "h_elo_pre", "a_elo_pre", "elo_home_prob"])
    out["elo_diff"] = out["h_elo_pre"] - out["a_elo_pre"]   # Home minus Away (no base HFA)
    return out, ratings


def build_perf_sched(sched, tg):
    """
    Copy of the schedule whose final scores are replaced by a PERFORMANCE margin: EPA_ELO_BLEND x the game's EPA
    margin (home offense EPA minus away offense EPA, garbage time removed) + the rest x the real score margin.
    Elo run on this learns from how teams played, not just the scoreboard (turnover luck, late touchdowns).
    """
    t = tg[["game_id", "is_home", "off_epa_tot"]]
    hm = t[t["is_home"] == 1].set_index("game_id")["off_epa_tot"]
    am = t[t["is_home"] == 0].set_index("game_id")["off_epa_tot"]
    out = sched.copy()
    em = out["game_id"].map(hm) - out["game_id"].map(am)             # EPA margin, home minus away
    sm = out["home_score"] - out["away_score"]                        # real score margin
    pm = (EPA_ELO_BLEND * em + (1 - EPA_ELO_BLEND) * sm).where(em.notna(), sm)
    played = out["home_score"].notna() & out["away_score"].notna()
    out["home_score"] = np.where(played, 100 + pm / 2, np.nan)
    out["away_score"] = np.where(played, 100 - pm / 2, np.nan)
    return out


def choose_elo_carryover(sched):
    """Score Elo alone with each candidate carryover on the TRAINING seasons and return the best (by log-loss)."""
    if ELO_SEASON_CARRYOVER is not None:
        return ELO_SEASON_CARRYOVER
    base = sched[["game_id", "season", "week", "result"]]
    out = []
    print("\n" + "=" * 78)
    print("ELO CARRYOVER TEST (Elo alone, training seasons only)")
    print("=" * 78)
    print(f"  {'carryover':>9s} {'log-loss':>9s} {'accuracy':>9s} {'wks 1-6 LL':>11s}")
    for c in ELO_CARRYOVER_CANDIDATES:
        el, _ = compute_elo(sched, c)
        m = base.merge(el[["game_id", "elo_home_prob"]], on="game_id")
        m = m[m["season"].isin(TUNE_SEASONS) & m["result"].notna() & (m["result"] != 0)]
        y = (m["result"] > 0).astype(float).values
        pr = np.clip(m["elo_home_prob"].values, 1e-6, 1 - 1e-6)
        ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
        early = (m["week"] <= 6).values
        out.append((ll.mean(), c))
        print(f"  {c:9.2f} {ll.mean():9.4f} {((pr >= 0.5) == y).mean():9.3f} {ll[early].mean():11.4f}")
    best = min(out)[1]
    print(f"  Using carryover {best:.2f} (set ELO_SEASON_CARRYOVER to override).")
    return best


def compute_points_ratings(sched, lr=0.08, carry=0.5, hfa=2.0):
    """
    Offense and defense points ratings. Each team has o (points it scores above average) and d (points its
    defense ALLOWS above average; lower is better). Before a game:
        home points = L + hfa/2 + o_home + d_away        away points = L - hfa/2 + o_away + d_home
    After the game each rating moves by lr x (actual points - projected points). Ratings regress toward 0 each
    new season. Returns a per-game frame: pr_margin (rating-only margin, no home edge) and pr_total (projected
    combined points), using only information from BEFORE the game.
    """
    games = sched.sort_values(["season", "week", "gameday", "game_id"])
    o, d, rows, cur = {}, {}, [], None
    tot_pts, tot_n = 0.0, 0
    for gid, season, ht, at, hs, as_, loc in zip(
            games["game_id"], games["season"], games["home_team"], games["away_team"],
            games["home_score"], games["away_score"], games["location"]):
        if season != cur:
            if cur is not None:
                for t in o:
                    o[t] *= carry
                for t in d:
                    d[t] *= carry
            cur = season
        L = tot_pts / tot_n if tot_n > 200 else 22.5
        h_ = 0.0 if str(loc).lower() == "neutral" else hfa
        oh, dh, oa, da = o.get(ht, 0.0), d.get(ht, 0.0), o.get(at, 0.0), d.get(at, 0.0)
        exp_h, exp_a = L + h_ / 2 + oh + da, L - h_ / 2 + oa + dh
        rows.append((gid, (oh + da) - (oa + dh), exp_h + exp_a))
        if pd.notna(hs) and pd.notna(as_):
            eh, ea = hs - exp_h, as_ - exp_a
            o[ht], d[at] = oh + lr * eh, da + lr * eh
            o[at], d[ht] = oa + lr * ea, dh + lr * ea
            tot_pts += hs + as_
            tot_n += 2
    return pd.DataFrame(rows, columns=["game_id", "pr_margin", "pr_total"])


def choose_points_params(sched):
    """Pick lr / carryover / home edge by margin error on the TUNING seasons only."""
    base = sched[["game_id", "season", "result"]]
    best = None
    for lr in PTS_LR_CANDIDATES:
        for c in PTS_CARRY_CANDIDATES:
            for h in PTS_HFA_CANDIDATES:
                pr = compute_points_ratings(sched, lr, c, h)
                m = base.merge(pr[["game_id", "pr_margin"]], on="game_id")
                m = m[m["season"].isin(TUNE_SEASONS) & m["result"].notna()]
                err = float(np.mean((m["result"] - (m["pr_margin"] + h)) ** 2))
                if best is None or err < best[0]:
                    best = (err, lr, c, h)
    err, lr, c, h = best
    print("\n" + "=" * 78)
    print("OFFENSE / DEFENSE POINTS RATINGS (tuned on the tuning seasons)")
    print("=" * 78)
    print(f"  Chosen: rating speed {lr:g}, season carryover {c:g}, home edge {h:g} pts "
          f"(margin error {np.sqrt(err):.2f} pts RMS)")
    return {"lr": lr, "carry": c, "hfa": h}


def add_continuity(tg, pbp):
    """
    Roster continuity (offense): the share of last season's offensive production (passes, runs and targets, by
    player) that came from players who are on the team's roster THIS season, plus a "new starting QB" flag
    (today's QB is not last season's main QB). Known before the season starts. Needs the rosters data; if that
    is unavailable the factor is neutral.
    """
    tg["cont"] = np.nan
    tg["newqb"] = 0.0
    try:
        d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]
        inv = pd.concat([d[["season", "posteam", c]].rename(columns={c: "pid"})
                         for c in ("passer_player_id", "rusher_player_id", "receiver_player_id")]).dropna(subset=["pid"])
        sh = inv.groupby(["season", "posteam", "pid"]).size().rename("n").reset_index()
        sh["share"] = sh["n"] / sh.groupby(["season", "posteam"])["n"].transform("sum")
        sh["season"] = sh["season"] + 1                                   # last season's share, indexed by this season
        sh = sh.rename(columns={"posteam": "team"})
        ro = to_pandas(nfl.load_rosters(SEASONS))
        id_c, tm_c, se_c = first_col(ro, ["gsis_id", "player_id"]), first_col(ro, ["team", "club_code", "team_abbr"]), first_col(ro, ["season"])
        if not all([id_c, tm_c, se_c]):
            raise ValueError(f"roster columns not found in {list(ro.columns)[:12]}")
        ro = ro[[id_c, tm_c, se_c]].rename(columns={id_c: "pid", tm_c: "team", se_c: "season"}).dropna().drop_duplicates()
        ro["team"] = fix_team(ro["team"])
        c = sh.merge(ro, on=["team", "season", "pid"]).groupby(["team", "season"])["share"].sum().rename("cont").reset_index()
        tg = tg.drop(columns=["cont"]).merge(c, on=["team", "season"], how="left")
        # seasons with no previous-season data (the first one loaded) stay NaN -> neutral below
    except Exception as ex:
        print(f"  ! roster continuity unavailable ({ex}); that factor will be neutral")
        tg["cont"] = np.nan
    mean_c = tg["cont"].mean()
    tg["cont"] = tg["cont"].fillna(mean_c if pd.notna(mean_c) else 0.0)
    try:
        q = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
        main_qb = (q.groupby(["season", "posteam", "passer_player_id"]).size().rename("n").reset_index()
                    .sort_values("n").drop_duplicates(["season", "posteam"], keep="last"))
        main_qb["season"] = main_qb["season"] + 1
        main_qb = main_qb.rename(columns={"posteam": "team", "passer_player_id": "prev_qb"})[["team", "season", "prev_qb"]]
        t2 = tg[["team", "season", "qb_id"]].merge(main_qb, on=["team", "season"], how="left")
        tg["newqb"] = ((t2["prev_qb"].notna()) & (t2["qb_id"].astype(object) != t2["prev_qb"].astype(object))
                       & t2["qb_id"].notna()).astype(float).values
    except Exception as ex:
        print(f"  ! new-QB flag unavailable ({ex})")
    return tg


def build_qb_deviation(tg):
    """
    Per game: how far today's QB is above/below the team's usual QB, in EPA/dropback (home = h_dev, away = a_dev).
    Usual = the team's average expected-QB quality over its previous 6 games (prior games only). A backup or a
    new starter shows up as a negative number for that team.
    """
    t = tg.sort_values(["team", "season", "week"]).copy()
    usual = t.groupby("team")["qb_q"].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    t["dev"] = (t["qb_q"] - usual).fillna(0.0).clip(-0.4, 0.4)
    h = t[t["is_home"] == 1].set_index("game_id")["dev"].rename("h_dev")
    a = t[t["is_home"] == 0].set_index("game_id")["dev"].rename("a_dev")
    return pd.concat([h, a], axis=1).fillna(0.0)


def choose_elo_params(sched, qb=None):
    """
    Pick the Elo settings. Always tests the season carryover; when ELO_AUTO_TUNE is on it also tunes K,
    home-field points, margin-of-victory scaling, an early-season K boost, the QB adjustment, and the rest /
    travel / team-home-field adjustments: a coarse grid first, then a finer one-setting-at-a-time search.
    Scored by Elo-only log-loss on the TRAINING seasons only. Returns compute_elo kwargs.
    """
    ctx = build_elo_context(sched)
    fixed = {"qb": qb, "ctx": ctx}
    if not ELO_AUTO_TUNE:
        return {"carryover": choose_elo_carryover(sched), "qb_scale": ELO_QB_SCALE, **fixed}
    base = sched[["game_id", "season", "week", "result"]]
    carries = [ELO_SEASON_CARRYOVER] if ELO_SEASON_CARRYOVER is not None else ELO_CARRYOVER_CANDIDATES
    cache = {}
    neutral = dict(carryover=0.0, k=ELO_K, hfa=ELO_HOME_ADV, mov=ELO_USE_MOV, early=ELO_EARLY_BOOST,
                   qb_scale=0.0, bye=0.0, short=0.0, travel=0.0, hfa_lr=0.0)

    def score(p):
        key = tuple(sorted((a_, round(float(v), 4)) for a_, v in p.items()))
        if key not in cache:
            el, _ = compute_elo(sched, **p, **fixed)
            m = base.merge(el[["game_id", "elo_home_prob"]], on="game_id")
            m = m[m["season"].isin(TUNE_SEASONS) & m["result"].notna() & (m["result"] != 0)]
            y = (m["result"] > 0).astype(float).values
            pr = np.clip(m["elo_home_prob"].values, 1e-6, 1 - 1e-6)
            ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
            cache[key] = (float(ll.mean()), float(((pr >= 0.5) == y).mean()))
        return cache[key]

    print("\n" + "=" * 78)
    print("ELO TUNING TEST (Elo alone, training seasons only)")
    print("=" * 78)
    cur_c = min(carries, key=lambda c: score({**neutral, "carryover": c})[0])
    cur = score({**neutral, "carryover": cur_c})
    grid = []
    for c in carries:                                           # stage 1: coarse grid
        for k in ELO_K_CANDIDATES:
            for h in ELO_HFA_CANDIDATES:
                for mv in ELO_MOV_CANDIDATES:
                    p = {**neutral, "carryover": c, "k": k, "hfa": h, "mov": mv}
                    grid.append(score(p) + (p,))
    grid.sort(key=lambda r: r[0])
    ll0, acc0, best_p = grid[0]
    print(f"  Stage 1 (coarse grid) best: carryover {best_p['carryover']:.2f}, K {best_p['k']:g}, "
          f"home {best_p['hfa']:g}, MOV {best_p['mov']}: log-loss {ll0:.4f}")
    best_p, best_ll = dict(best_p), ll0
    steps = [("k", lambda v: [v * f for f in (0.6, 0.8, 0.9, 1.1, 1.25, 1.5)]),
             ("hfa", lambda v: [v + d for d in (-15, -8, -4, 4, 8, 15)]),
             ("carryover", lambda v: [min(max(v + d, 0.0), 1.0) for d in (-0.15, -0.08, -0.04, 0.04, 0.08, 0.15)]),
             ("early", lambda v: ELO_EARLY_BOOST_CANDIDATES),
             ("qb_scale", lambda v: ELO_QB_SCALE_CANDIDATES if qb is not None else []),
             ("bye", lambda v: ELO_BYE_CANDIDATES),
             ("short", lambda v: ELO_SHORT_CANDIDATES),
             ("travel", lambda v: ELO_TRAVEL_CANDIDATES),
             ("hfa_lr", lambda v: ELO_HFA_LR_CANDIDATES)]
    for _ in range(3):                                          # stage 2: refine one setting at a time
        improved = False
        for name, cands in steps:
            for v in cands(best_p[name]):
                t = {**best_p, name: v}
                ll = score(t)[0]
                if ll < best_ll - 1e-5:
                    best_p, best_ll, improved = t, ll, True
        if not improved:
            break
    acc = score(best_p)[1]
    no_extras = {**best_p, "qb_scale": 0.0, "bye": 0.0, "short": 0.0, "travel": 0.0, "hfa_lr": 0.0}
    ll_ne, acc_ne = score(no_extras)
    print(f"  {'log-loss':>9s} {'accuracy':>9s}  settings")
    print(f"  {cur[0]:9.4f} {cur[1]:9.3f}  your old defaults (K {ELO_K:g}, home {ELO_HOME_ADV:g}, MOV {ELO_USE_MOV}, "
          f"carryover {cur_c:.2f})")
    print(f"  {ll0:9.4f} {acc0:9.3f}  after the coarse grid")
    print(f"  {ll_ne:9.4f} {acc_ne:9.3f}  fine-tuned base settings, no QB / rest / travel / team-HFA adjustments")
    print(f"  {best_ll:9.4f} {acc:9.3f}  final, with those adjustments")
    p = best_p
    print(f"  Chosen: carryover {p['carryover']:.2f}, K {p['k']:.1f}, home-field {p['hfa']:.0f} pts, MOV {p['mov']}, "
          f"early-season K x{p['early']:g}")
    print(f"          QB {p['qb_scale']:g} Elo pts per 1.0 EPA/dropback | bye/long rest +{p['bye']:g} | "
          f"short week -{p['short']:g} | travel {p['travel']:g} per 1,000 mi | dynamic team home edge rate {p['hfa_lr']:g}")
    print("  (A value of 0 means that adjustment did not help and is off. Set the matching *_CANDIDATES to [0.0] "
          "to force it off, or ELO_AUTO_TUNE = False for the fixed values.)")
    return {**best_p, **fixed}


def print_elo_rankings(ratings):
    s = pd.Series(ratings).sort_values(ascending=False)
    print("\n" + "=" * 78)
    print(f"CURRENT ELO RATINGS (after all completed games, {PREDICT_SEASON} season)")
    print("=" * 78)
    for i, (team, r) in enumerate(s.items(), 1):
        print(f"{i:>2}. {team:<4s} {r:7.1f}")


# ----------------------------------------------------------------------------
# 5c. TEAM-SPECIFIC HOME-FIELD ADVANTAGE
# ----------------------------------------------------------------------------
def add_team_hfa(tg):
    """
    Team-specific home-field advantage (in points), using PRIOR games only.

        raw_hfa = (avg home margin - avg road margin) / 2
        hfa_pre = shrink raw_hfa toward LEAGUE_HFA_PTS with weight HFA_SHRINK_GAMES

    A team with no history gets exactly the league value. For an AWAY team the
    same number also describes how much worse it is on the road, so the matchup
    feature is (home team's hfa_pre - away team's hfa_pre).
    """
    margin = tg["pts_for"] - tg["pts_against"]
    played = margin.notna()
    home, away = played & (tg["is_home"] == 1), played & (tg["is_home"] == 0)
    tmp = pd.DataFrame({"team": tg["team"],
                        "hm": margin.where(home, 0.0), "am": margin.where(away, 0.0),
                        "nh": home.astype(int), "na": away.astype(int)})
    cum = tmp.groupby("team")[["hm", "am", "nh", "na"]].cumsum()
    prev = cum.groupby(tmp["team"]).shift(1).fillna(0.0)          # totals BEFORE this game
    avg_h = prev["hm"] / prev["nh"].where(prev["nh"] > 0)
    avg_a = prev["am"] / prev["na"].where(prev["na"] > 0)
    raw = ((avg_h - avg_a) / 2).fillna(0.0)
    n = np.minimum(prev["nh"], prev["na"])
    tg["hfa_pre"] = (raw * n + LEAGUE_HFA_PTS * HFA_SHRINK_GAMES) / (n + HFA_SHRINK_GAMES)
    return tg


# ----------------------------------------------------------------------------
# 6. PRE-GAME INJURY IMPACT
# ----------------------------------------------------------------------------
def prior_leader(counts, tg, window, exclude=None):
    """
    For each (team, gnum) return the player id with the highest total `cnt`
    over that team's previous `window` games (strictly before this game).
    `exclude` (optional) = one list of player ids per tg row to skip (for WR2 / WR3 / backup QB).
    """
    by_team = {t: d for t, d in counts.groupby("team")}
    out = []
    for i, (team, gnum) in enumerate(zip(tg["team"], tg["gnum"])):
        d = by_team.get(team)
        if d is None:
            out.append(None)
            continue
        w = d[(d["gnum"] < gnum) & (d["gnum"] >= gnum - window)]
        tot = w.groupby("pid")["cnt"].sum()
        if exclude is not None and len(tot):
            tot = tot.drop([x for x in exclude[i] if x in tot.index])
        out.append(tot.idxmax() if len(tot) else None)
    return out


def add_starter_ids(tg, pbp, pos_map, depth_qb1):
    """Expected starting QB, WR1 and RB1 per team-game using only PRIOR games."""
    gmap = tg[["game_id", "team", "gnum"]]

    def counts(df, team_col, pid_col, positions=None):
        d = df[[ "game_id", team_col, pid_col]].dropna().rename(columns={team_col: "team", pid_col: "pid"})
        if positions is not None and pos_map:
            d = d[d["pid"].map(pos_map).isin(positions)]
        d = d.merge(gmap, on=["game_id", "team"], how="inner")
        return d.groupby(["team", "gnum", "pid"]).size().rename("cnt").reset_index()

    db = pbp[(pbp["qb_dropback"] == 1) & pbp["play_type"].isin(["pass", "run"])].copy()
    db["qb_id"] = db["passer_player_id"].fillna(db["rusher_player_id"])
    qb_counts = counts(db, "posteam", "qb_id")

    tgt = pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"].fillna(0) != 1)]
    wr_counts = counts(tgt, "posteam", "receiver_player_id", positions={"WR"})

    rush = pbp[(pbp["play_type"] == "run") & (pbp["qb_scramble"].fillna(0) != 1)]
    rb_counts = counts(rush, "posteam", "rusher_player_id", positions={"RB", "FB", "HB"})

    tg["qb_id"] = prior_leader(qb_counts, tg, window=3)       # most dropbacks, last 3 games
    tg["wr1_id"] = prior_leader(wr_counts, tg, window=6)      # most targets, last 6 games
    tg["rb1_id"] = prior_leader(rb_counts, tg, window=6)      # most carries, last 6 games
    # backup QB = next-most dropbacks over the last 17 games; WR2 / WR3 = next-most targets
    tg["qb2_id"] = prior_leader(qb_counts, tg, window=17, exclude=[[q] for q in tg["qb_id"]])
    tg["wr2_id"] = prior_leader(wr_counts, tg, window=6, exclude=[[a] for a in tg["wr1_id"]])
    tg["wr3_id"] = prior_leader(wr_counts, tg, window=6,
                                exclude=[[a, b] for a, b in zip(tg["wr1_id"], tg["wr2_id"])])

    # depth-chart QB1 only fills gaps (e.g. 2022 Week 1 has no prior games)
    fallback = [depth_qb1.get((t, s, w)) for t, s, w in zip(tg["team"], tg["season"], tg["week"])]
    tg["qb_id"] = tg["qb_id"].where(tg["qb_id"].notna(), pd.Series(fallback, index=tg.index))
    return tg


def qb_quality_table(pbp):
    """Cumulative EPA/dropback per QB by (season, week), used with a strict-before lookup."""
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["play_type"].isin(["pass", "run"])].copy()
    d["pid"] = d["passer_player_id"].fillna(d["rusher_player_id"])
    d = d.dropna(subset=["pid", "epa"])
    if d.empty:
        return None
    d["t"] = d["season"].astype(int) * 100 + d["week"].astype(int)
    g = d.groupby(["pid", "t"]).agg(e=("epa", "sum"), n=("epa", "size")).reset_index().sort_values(["pid", "t"])
    g["cum_e"] = g.groupby("pid")["e"].cumsum()
    g["cum_n"] = g.groupby("pid")["n"].cumsum()
    g["pid"] = g["pid"].astype(object)
    return g[["pid", "t", "cum_e", "cum_n"]].sort_values("t")


QB_PR_PRIOR_ATT = 150          # passer rating is shrunk toward the league average, worth this many attempts
LEAGUE_QB_PR = (0.64, 7.1, 0.045, 0.022)    # completion %, yards/attempt, TD rate, INT rate (league-average QB)


def _passer_rating(comp, ypa, td, ints):
    r = lambda x: np.clip(x, 0, 2.375)
    return (r((comp - 0.3) * 5) + r((ypa - 3) * 0.25) + r(td * 20) + r(2.375 - ints * 25)) / 6 * 100


def qb_prate_table(pbp):
    """Cumulative passing totals per QB by (season, week), used with a strict-before lookup."""
    d = pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"].fillna(0) != 1)
            & (pbp["two_point_attempt"].fillna(0) != 1)].copy()
    d = d.dropna(subset=["passer_player_id"])
    if d.empty or d["complete_pass"].notna().sum() == 0:
        return None
    d["pid"] = d["passer_player_id"]
    d["t"] = d["season"].astype(int) * 100 + d["week"].astype(int)
    d["c"] = d["complete_pass"].fillna(0)
    d["y"] = d["yards_gained"].fillna(0)
    d["td"] = (d["pass_touchdown"].fillna(0) == 1).astype(int)
    d["i"] = (d["interception"].fillna(0) == 1).astype(int)
    d["a"] = 1
    g = d.groupby(["pid", "t"])[["c", "y", "td", "i", "a"]].sum().reset_index().sort_values(["pid", "t"])
    for c in ["c", "y", "td", "i", "a"]:
        g["cum_" + c] = g.groupby("pid")[c].cumsum()
    g["pid"] = g["pid"].astype(object)
    return g[["pid", "t"] + ["cum_" + c for c in ["c", "y", "td", "i", "a"]]].sort_values("t")


def qb_prate_at(table, pids, ts):
    """Shrunk passer rating for each (pid, t), using only games BEFORE week t (unknown QB = league average)."""
    lc, ly, lt, li = LEAGUE_QB_PR
    league = _passer_rating(lc, ly, lt, li)
    q = pd.DataFrame({"pid": pd.Series(list(pids), dtype=object), "t": list(ts)})
    q["i"] = np.arange(len(q))
    if table is None or q.empty:
        return np.full(len(q), league)
    ok = q.dropna(subset=["pid"]).sort_values("t")
    m = pd.merge_asof(ok, table, on="t", by="pid", allow_exact_matches=False)
    for c in ["cum_c", "cum_y", "cum_td", "cum_i", "cum_a"]:
        m[c] = m[c].fillna(0.0)
    n = m["cum_a"] + QB_PR_PRIOR_ATT
    val = _passer_rating((m["cum_c"] + lc * QB_PR_PRIOR_ATT) / n, (m["cum_y"] + ly * QB_PR_PRIOR_ATT) / n,
                         (m["cum_td"] + lt * QB_PR_PRIOR_ATT) / n, (m["cum_i"] + li * QB_PR_PRIOR_ATT) / n)
    out = pd.Series(league, index=q["i"])
    out.loc[m["i"].values] = val.values
    return out.sort_index().values


def qb_quality_at(table, pids, ts):
    """Shrunk EPA/dropback for each (pid, t), using only games BEFORE week t."""
    q = pd.DataFrame({"pid": pd.Series(list(pids), dtype=object), "t": list(ts)})
    q["i"] = np.arange(len(q))
    if table is None or q.empty:
        return np.full(len(q), QB_PRIOR_EPA)
    ok = q.dropna(subset=["pid"]).sort_values("t")
    m = pd.merge_asof(ok, table, on="t", by="pid", allow_exact_matches=False)
    m["cum_e"] = m["cum_e"].fillna(0.0)
    m["cum_n"] = m["cum_n"].fillna(0.0)
    val = (m["cum_e"] + QB_PRIOR_DROPBACKS * QB_PRIOR_EPA) / (m["cum_n"] + QB_PRIOR_DROPBACKS)
    out = pd.Series(QB_PRIOR_EPA, index=q["i"])
    out.loc[m["i"].values] = val.values
    return out.sort_index().values


def load_snap_share(pos_pfr_map=None):
    """Rolling 6-game offensive snap share per player (gsis_id), or None if unavailable."""
    try:
        sn = to_pandas(nfl.load_snap_counts(SEASONS))
        pl = to_pandas(nfl.load_players())
        pfr = dict(zip(pl["pfr_id"], pl["gsis_id"]))
        pid_c = first_col(sn, ["pfr_player_id", "pfr_id"])
        pct_c = first_col(sn, ["offense_pct", "off_pct"])
        dpct_c = first_col(sn, ["defense_pct", "def_pct"])
        if pid_c is None or pct_c is None:
            raise ValueError("snap count columns not found")
        if "game_type" in sn.columns:
            sn = sn[sn["game_type"] == "REG"]
        pc = pd.to_numeric(sn[pct_c], errors="coerce")
        if dpct_c is not None:                      # a defender's share is his DEFENSIVE snap share
            pc = np.fmax(pc.fillna(0.0), pd.to_numeric(sn[dpct_c], errors="coerce").fillna(0.0))
        sn = sn.assign(gsis_id=sn[pid_c].map(pfr), pct=pc).dropna(subset=["gsis_id", "pct"])
        if sn["pct"].max() > 1.5:
            sn["pct"] = sn["pct"] / 100.0
        sn["t"] = sn["season"].astype(int) * 100 + sn["week"].astype(int)
        sn = sn.sort_values(["gsis_id", "t"])
        sn["snap6"] = sn.groupby("gsis_id")["pct"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        return sn[["gsis_id", "t", "snap6"]].sort_values("t") if len(sn) else None
    except Exception as e:
        print(f"  ! snap counts unavailable ({e}); non-RB1 backs will count at a flat {RB_NO_SNAP_DATA_WEIGHT}")
        return None


RUSH_CLASS = {"DE": "edge", "OLB": "edge", "EDGE": "edge", "DT": "int", "NT": "int", "DL": "int",
              "LB": "lb", "ILB": "lb", "MLB": "lb"}


def load_rush_quality():
    """
    Production multiplier for pass rushers and linebackers: each player's average QB pressures per game over his last 12
    games (PFR advanced defense), relative to a typical starter at his position group, shrunk toward 1 and limited to
    0.7-1.8. Returns a table (gsis_id, t, rq) to be looked up as of before a given week, or None if the data is missing.
    """
    try:
        d = to_pandas(nfl.load_pfr_advstats(SEASONS, stat_type="def", summary_level="week"))
        pl = to_pandas(nfl.load_players())
        pid = first_col(d, ["pfr_player_id", "pfr_id"])
        pos_c = first_col(pl, ["position", "pos"])
        if pid is None or pos_c is None or not {"season", "week"} <= set(d.columns):
            raise ValueError("columns not as expected")
        pc = first_col(d, ["def_pressures"])
        if pc is not None:
            d["pr_"] = pd.to_numeric(d[pc], errors="coerce")
        else:
            parts = [pd.to_numeric(d[c], errors="coerce").fillna(0) for c in ("def_times_hurried", "def_times_hitqb", "def_sacks") if c in d.columns]
            if not parts:
                raise ValueError("no pressure columns")
            d["pr_"] = sum(parts)
        d["gsis_id"] = d[pid].map(dict(zip(pl["pfr_id"], pl["gsis_id"])))
        d["cls"] = d["gsis_id"].map(dict(zip(pl["gsis_id"], pl[pos_c].astype(str).str.upper()))).map(RUSH_CLASS)
        d = d.dropna(subset=["gsis_id", "cls", "pr_"])
        d["t"] = d["season"].astype(int) * 100 + d["week"].astype(int)
        d = d.sort_values(["gsis_id", "t"])
        d["roll"] = d.groupby("gsis_id")["pr_"].transform(lambda x: x.rolling(12, min_periods=3).mean())
        d = d.dropna(subset=["roll"])
        if d.empty:
            return None
        base = d.groupby("cls")["roll"].transform(lambda x: x[x >= x.median()].mean())      # a typical starter in the group
        d["rq"] = np.clip(1.0 + 0.6 * (d["roll"] / base.where(base > 0) - 1.0), 0.7, 1.8).fillna(1.0)
        return d[["gsis_id", "t", "rq"]].sort_values("t")
    except Exception as ex:
        print(f"  ! pass-rush production unavailable ({ex}); missing pass rushers count by position and snaps only")
        return None


def load_target_quality(pbp):
    """
    Volume multiplier for receivers: each WR / TE's targets per game over his last 12 games, relative to a typical starter
    at his position, shrunk toward 1 and limited to 0.7-1.8. Table (gsis_id, t, rq) looked up as of before a given week.
    """
    try:
        d = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()]
        d = d.groupby(["receiver_player_id", "posteam", "season", "week"]).size().rename("tgt").reset_index()
        d = d.rename(columns={"receiver_player_id": "gsis_id"})
        d["share_g"] = d["tgt"] / d.groupby(["posteam", "season", "week"])["tgt"].transform("sum")   # share of the team's targets that game
        d["t"] = d["season"].astype(int) * 100 + d["week"].astype(int)
        d = d.sort_values(["gsis_id", "t"])
        d["roll"] = d.groupby("gsis_id")["tgt"].transform(lambda x: x.rolling(12, min_periods=3).mean())
        d["share"] = d.groupby("gsis_id")["share_g"].transform(lambda x: x.rolling(12, min_periods=3).mean())
        d = d.dropna(subset=["roll"])
        cls = pd.Series("wr", index=d.index)
        try:
            pl = to_pandas(nfl.load_players())
            pos_c = first_col(pl, ["position", "pos"])
            if pos_c is not None:
                cls = d["gsis_id"].map(dict(zip(pl["gsis_id"], pl[pos_c].astype(str).str.upper()))).eq("TE").map({True: "te", False: "wr"})
        except Exception:
            pass
        d["cls"] = cls
        base = d.groupby("cls")["roll"].transform(lambda x: x[x >= x.median()].mean())
        d["rq"] = np.clip(1.0 + 0.6 * (d["roll"] / base.where(base > 0) - 1.0), 0.7, 1.8).fillna(1.0)
        return d[["gsis_id", "t", "rq", "share"]].sort_values("t")
    except Exception as ex:
        print(f"  ! receiver target volumes unavailable ({ex}); missing receivers count by depth chart only")
        return None


def load_cb_quality():
    """
    Coverage multiplier for corners: yards allowed per target over his last 12 games (PFR), shrunk toward the league rate
    (30 targets of prior), turned into a multiplier (better coverage = bigger loss when he is out), limited to 0.8-1.4.
    """
    try:
        d = to_pandas(nfl.load_pfr_advstats(SEASONS, stat_type="def", summary_level="week"))
        pl = to_pandas(nfl.load_players())
        pid = first_col(d, ["pfr_player_id", "pfr_id"])
        if pid is None or "def_targets" not in d.columns or "def_yards_allowed" not in d.columns:
            raise ValueError("columns not as expected")
        d["gsis_id"] = d[pid].map(dict(zip(pl["pfr_id"], pl["gsis_id"])))
        d = d.dropna(subset=["gsis_id"])
        d["tg_"] = pd.to_numeric(d["def_targets"], errors="coerce").fillna(0.0)
        d["yd_"] = pd.to_numeric(d["def_yards_allowed"], errors="coerce").fillna(0.0)
        d = d[d["tg_"] > 0]
        d["t"] = d["season"].astype(int) * 100 + d["week"].astype(int)
        d = d.sort_values(["gsis_id", "t"])
        lg = d["yd_"].sum() / max(d["tg_"].sum(), 1.0)
        g = d.groupby("gsis_id")
        ty = g["tg_"].transform(lambda x: x.rolling(12, min_periods=3).sum())
        yy = g["yd_"].transform(lambda x: x.rolling(12, min_periods=3).sum())
        d["ypt"] = (yy + 30 * lg) / (ty + 30)
        d = d.dropna(subset=["ypt"])
        d["rq"] = np.clip(1.0 + 0.6 * (lg / d["ypt"] - 1.0) * 2.0, 0.8, 1.4)
        return d[["gsis_id", "t", "rq"]].sort_values("t")
    except Exception as ex:
        print(f"  ! corner coverage stats unavailable ({ex}); missing corners count by position and snaps only")
        return None


def add_qb_quality(tg, pbp):
    """
    qb_q = pre-game EPA/dropback of the QB expected to play (shrunk toward QB_PRIOR_EPA). If the
    starter is Out/Doubtful this week, the backup's number is used instead. Prior games only.
    """
    tbl = qb_quality_table(pbp)
    t = tg["season"].astype(int) * 100 + tg["week"].astype(int)
    q1 = qb_quality_at(tbl, tg["qb_id"], t)
    q2 = qb_quality_at(tbl, tg["qb2_id"], t)
    tg["qb_q"] = np.where(tg["qb_out"].fillna(0) == 1, q2, q1)
    pt = qb_prate_table(pbp)
    if pt is None:
        print("  ! no completion data in the play-by-play; QB passer rating will be league average for everyone")
    p1 = qb_prate_at(pt, tg["qb_id"], t)
    p2 = qb_prate_at(pt, tg["qb2_id"], t)
    tg["qb_pr"] = np.where(tg["qb_out"].fillna(0) == 1, p2, p1)       # the QB expected to play (backup if out)
    return tg


def manual_injury_rows(tgt):
    """Rows (same columns as the injury feed) for MANUAL_INJURIES entries of week tgt, with ids looked up by name."""
    ent = MANUAL_INJURIES.get(tgt) if tgt is not None else None
    if not ent:
        if MANUAL_INJURIES and tgt is not None:
            print(f"  Manual injuries are set for week(s) {sorted(MANUAL_INJURIES)}, not for week {tgt}; none applied.")
        return pd.DataFrame()
    try:
        pl = to_pandas(nfl.load_players())
    except Exception as ex:
        print(f"  ! manual injuries not applied (player list unavailable: {ex})")
        return pd.DataFrame()
    nm_c = first_col(pl, ["display_name", "full_name", "player_name", "name"])
    pos_c = first_col(pl, ["position", "pos"])
    id_c = first_col(pl, ["gsis_id"])
    if nm_c is None or pos_c is None or id_c is None:
        print("  ! manual injuries not applied (player list columns not recognized)")
        return pd.DataFrame()
    key = pl[nm_c].astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    rows = []
    for team, plist in ent.items():
        tm_ = fix_team(pd.Series([team])).iloc[0]
        for name, status in plist:
            k = re.sub(r"[^a-z ]", "", name.lower()).strip()
            m = pl[(key == k) & pl[id_c].notna()]
            if len(m) > 1 and "status" in pl.columns:
                m = m[m["status"].astype(str).str.upper().isin(["ACT", "ACTIVE"])] if (m["status"].astype(str).str.upper().isin(["ACT", "ACTIVE"])).any() else m
            if m.empty:
                print(f"  ! manual injury: could not find '{name}' in the player list; skipped")
                continue
            m = m.iloc[-1]
            rows.append(dict(season=PREDICT_SEASON, season_type="REG", game_type="REG", team=tm_, week=tgt, gsis_id=m[id_c],
                             position=str(m[pos_c]).upper(), full_name=m[nm_c], report_status=status.title(), game_status=status.title(),
                             report_primary_injury="(added by hand)", practice_status="Did Not Participate In Practice"))
    if rows:
        print(f"  Manual injuries applied for week {tgt}: " + ", ".join(f"{r['full_name']} ({r['team']}, {r['report_status']})" for r in rows))
    return pd.DataFrame(rows)


def add_manual_injuries(inj, tgt):
    """Add the manual rows for week tgt to an injury frame, replacing any feed row for the same player and week."""
    man = manual_injury_rows(tgt)
    if man.empty:
        return inj
    drop = inj.set_index(["gsis_id", "week"]).index.isin(list(zip(man["gsis_id"], man["week"])))
    return pd.concat([inj[~drop], man], ignore_index=True)


def latest_team_reports(df, tgt, max_age=3):
    """Each team's most recent injury report (rows with the team's own latest week), if it is at most max_age weeks old.
    A team that just had a bye has nothing in the league's latest week, so it falls back to the report from before the bye."""
    if df.empty:
        return df
    mx = df.groupby("team")["week"].transform("max")
    return df[(df["week"] == mx) & (tgt - mx <= max_age)]


def build_injury_table(tg, pbp=None):
    """
    Per (team, season, week): qb_out flag + weighted key-position injury score.
    Only 'Out' / 'Doubtful' players count. No shift(1): the report precedes kickoff.
    """
    empty = pd.DataFrame(columns=["team", "season", "week", "qb_out", "inj_score", "wr_lost"] + INJ_GROUP_COLS)
    try:
        inj = to_pandas(nfl.load_injuries(SEASONS))
    except Exception as e:
        print(f"  ! load_injuries failed ({e}); injury features will be 0")
        return empty

    status_c = first_col(inj, ["report_status", "game_status"])
    if status_c is None or "gsis_id" not in inj.columns:
        print("  ! injuries schema unexpected; injury features will be 0")
        return empty
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    inj = inj.copy()
    inj["team"] = fix_team(inj["team"])
    inj["status"] = inj[status_c].astype(str).str.strip().str.title()
    if INJURY_CARRY_LAST_WEEK:
        cur_s = inj[inj["season"] == PREDICT_SEASON]
        tgt = pick_week(tg[tg["season"] == PREDICT_SEASON][["season", "week", "pts_for"]].rename(columns={"pts_for": "result"}))
        if len(cur_s) and tgt is not None and int(cur_s["week"].max()) < tgt:
            last_w = int(cur_s["week"].max())
            dup = latest_team_reports(cur_s, tgt).copy()        # each team's own latest report (covers teams coming off a bye)
            n_old = int(dup.loc[dup["week"] < last_w, "team"].nunique())
            dup["week"] = tgt
            inj = pd.concat([inj, dup], ignore_index=True)
            globals()["INJ_CARRIED_FROM"] = last_w
            print(f"  ! TEST MODE: using the week {last_w} injury report as the week {tgt} report (INJURY_CARRY_LAST_WEEK)"
                  + (f"; {n_old} team(s) had no report that week (bye), so their earlier one is used." if n_old else "."))
    tgt_ = pick_week(tg[tg["season"] == PREDICT_SEASON][["season", "week", "pts_for"]].rename(columns={"pts_for": "result"}))
    if "season_type" not in inj.columns:
        inj["season_type"] = "REG"
    inj = add_manual_injuries(inj, tgt_)
    inj["status"] = inj[status_c].astype(str).str.strip().str.title()
    nm_c = first_col(inj, ["full_name", "player_name", "name"])
    if nm_c is not None:
        inj["pname"] = inj[nm_c].astype(str)
    elif {"first_name", "last_name"} <= set(inj.columns):
        inj["pname"] = inj["first_name"].fillna("") + " " + inj["last_name"].fillna("")
    else:
        inj["pname"] = ""
    inj = inj[inj["status"].isin(STATUS_FACTOR)].copy()
    inj["position"] = inj["position"].astype(str).str.upper().str.strip()
    # one row per player-week: keep the more severe status
    inj["sev"] = inj["status"].map(STATUS_FACTOR)
    inj = (inj.sort_values("sev", ascending=False)
              .drop_duplicates(["team", "season", "week", "gsis_id"]))

    ids = tg[["team", "season", "week", "qb_id", "qb2_id", "wr1_id", "wr2_id", "wr3_id", "rb1_id"]]
    inj = inj.merge(ids, on=["team", "season", "week"], how="left")
    inj["t"] = inj["season"].astype(int) * 100 + inj["week"].astype(int)

    # starting-QB penalty scales with how much worse the backup is
    inj["qb_mult"] = 1.0
    if pbp is not None:
        tbl = qb_quality_table(pbp)
        q_start = qb_quality_at(tbl, inj["qb_id"], inj["t"])
        q_back = qb_quality_at(tbl, inj["qb2_id"], inj["t"])
        inj["qb_mult"] = np.clip((q_start - q_back) / QB_DROP_SCALE, QB_DROP_MIN, QB_DROP_MAX)

    # backup-RB snap share (prior 6 games); NaN if never seen / no data
    snaps = load_snap_share()
    have_snaps = snaps is not None
    inj["snap6"] = np.nan
    if have_snaps:
        o = inj.dropna(subset=["gsis_id"]).sort_values("t")[["gsis_id", "t"]].copy()
        o["i"] = o.index
        m = pd.merge_asof(o, snaps, on="t", by="gsis_id", allow_exact_matches=False)
        inj.loc[m["i"].values, "snap6"] = m["snap6"].values

    inj["rq"] = np.nan
    inj["wq"] = np.nan
    inj["cq"] = np.nan
    inj["wshare"] = np.nan
    for col, loader, on_ in (("rq", load_rush_quality, INJURY_STAR_QUALITY),
                             ("wq", (lambda: load_target_quality(pbp)) if pbp is not None else (lambda: None), INJURY_WR_QUALITY),
                             ("cq", load_cb_quality, INJURY_CB_QUALITY)):
        tbl_ = loader() if (INJURY_PLAYER_IMPACT and on_) else None
        if tbl_ is not None:
            o = inj.dropna(subset=["gsis_id"]).sort_values("t")[["gsis_id", "t"]].copy()
            o["i"] = o.index
            m = pd.merge_asof(o, tbl_, on="t", by="gsis_id", allow_exact_matches=False)
            inj.loc[m["i"].values, col] = m["rq"].values
            if col == "wq" and "share" in m.columns:
                inj.loc[m["i"].values, "wshare"] = m["share"].values

    def weight(r):
        pos, pid = r["position"], r["gsis_id"]
        if pos == "QB":                       # only the expected STARTER counts, scaled by the backup gap
            return QB_WEIGHT * r["qb_mult"] if pid == r["qb_id"] else 0.0
        if pos == "WR":
            wm = 1.0 if pd.isna(r["wq"]) else float(r["wq"])      # heavily targeted receivers count more
            if pid == r["wr1_id"]:
                return KEY_SKILL_WEIGHT * wm
            if pid == r["wr2_id"]:
                return WR2_WEIGHT * wm
            if pid == r["wr3_id"]:
                return WR3_WEIGHT * wm
            return WR_DEPTH_WEIGHT * wm
        if pos in ("RB", "FB", "HB"):
            if pid == r["rb1_id"]:
                return KEY_SKILL_WEIGHT
            if not have_snaps:
                return RB_NO_SNAP_DATA_WEIGHT
            return RB_BACKUP_WEIGHT if r["snap6"] >= RB_BACKUP_SNAP_MIN else 0.0   # NaN -> 0
        base = POS_WEIGHTS.get(pos, DEPTH_WEIGHT)
        if INJURY_PLAYER_IMPACT and have_snaps and pos not in ("K", "P", "LS"):
            sp = r["snap6"]                      # how much he really plays: ~15% = depth (0), 65%+ = a full starter (1)
            f = 0.35 if pd.isna(sp) else float(np.clip((sp - 0.15) / 0.5, 0.0, 1.0))
            if pos in RUSH_CLASS and pd.notna(r["rq"]):
                f *= float(r["rq"])              # productive pass rushers / linebackers count more
            if pos == "TE" and pd.notna(r["wq"]):
                f *= float(r["wq"])              # heavily targeted tight ends count more
            if pos == "CB" and pd.notna(r["cq"]):
                f *= float(r["cq"])              # corners who have covered well count more
            return base * f
        return base

    inj["w"] = inj.apply(weight, axis=1) * inj["sev"]
    # remember this week's missing players (for names on the game cards and the impact column in the injury report)
    INJ_KEY_NOW.clear()
    INJ_IMPACT_NOW.clear()
    cur = inj[inj["season"] == PREDICT_SEASON]
    if len(cur):
        cur = cur[cur["week"] == cur["week"].max()]
        globals()["INJ_WEEK_NOW"] = int(cur["week"].max())
        print(f"  Injury data: latest week found is {INJ_WEEK_NOW} ({cur['team'].nunique()} teams have a listed player).")
        for _, x in cur.iterrows():
            INJ_IMPACT_NOW[(x["team"], x["gsis_id"])] = float(x["w"])
            if x["w"] >= KEY_PLAYER_W and x["pname"]:
                INJ_KEY_NOW.setdefault(x["team"], []).append((float(x["w"]), x["pname"], x["position"], x["status"]))
        for t_ in INJ_KEY_NOW:
            INJ_KEY_NOW[t_].sort(key=lambda z: -z[0])
    inj["qb_flag"] = ((inj["position"] == "QB") & (inj["gsis_id"] == inj["qb_id"])).astype(int)
    pos_group = {p: g for g, ps in INJ_GROUPS.items() for p in ps}
    inj["grp"] = inj["position"].map(pos_group)
    for g in INJ_GROUPS:
        inj[f"inj_{g}"] = np.where(inj["grp"] == g, inj["w"], 0.0)
    inj["lost_share"] = np.where(inj["position"].isin(["WR", "TE"]), inj["wshare"].fillna(0.0) * inj["sev"], 0.0)
    agg = (inj.groupby(["team", "season", "week"])
              .agg(qb_out=("qb_flag", "max"), inj_score=("w", "sum"), wr_lost=("lost_share", "sum"),
                   **{c: (c, "sum") for c in INJ_GROUP_COLS}).reset_index())
    return agg


# ----------------------------------------------------------------------------
# 7. MATCHUP FEATURES
# ----------------------------------------------------------------------------
TEAM_FEATS = ["anya_off_pre", "anya_def_pre", "epa_off_pre", "epa_def_pre", "net_epa_pre",
              "sos", "sos_adj_net_epa", "qb_out", "inj_score", "wr_lost", "hfa_pre"] + INJ_GROUP_COLS + [
              "rz_off_pre", "rz_def_pre", "tov_margin_pre", "expl_off_pre", "expl_def_pre",
              "succ_off_pre", "succ_def_pre", "off_unit", "def_unit", "off_rank", "def_rank",
              "qb_q", "qb_pr", "plays_pre", "go_pre", "cont", "newqb",
              "sty_pass_off_pre", "sty_pass_def_pre", "sty_rush_off_pre", "sty_rush_def_pre",
              "sty_press_off_pre", "sty_press_def_pre"] + OFF_UNIT_COLS + DEF_UNIT_COLS


# Combined (home + away) scoring-environment features for the projected-total model
TOTAL_BASE = ["anya_off_pre", "anya_def_pre", "epa_off_pre", "epa_def_pre",
              "rz_off_pre", "rz_def_pre", "expl_off_pre", "expl_def_pre", "plays_pre", "sty_pass_off_pre", "sty_rush_off_pre"]
TOTAL_FEATS = ["tot_" + k for k in TOTAL_BASE] + ["pr_total", "wx_wind", "wx_cold"]


def build_matchups(sched, tg, elo):
    h = tg[tg["is_home"] == 1][["game_id"] + TEAM_FEATS].rename(
        columns={c: "h_" + c for c in TEAM_FEATS})
    a = tg[tg["is_home"] == 0][["game_id"] + TEAM_FEATS].rename(
        columns={c: "a_" + c for c in TEAM_FEATS})
    g = (sched[["game_id", "season", "week", "gameday", "home_team", "away_team",
                "home_score", "away_score", "result", "location"]
               + [c for c in ("gametime",) if c in sched.columns]
               + [c for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline", "roof", "temp", "wind")
                  if c in sched.columns]]
         .merge(h, on="game_id").merge(a, on="game_id"))

    # Home minus Away (for "defense" columns, LOWER is better, so negative favors home)
    g["anya_off_diff"] = g["h_anya_off_pre"] - g["a_anya_off_pre"]
    g["anya_def_diff"] = g["h_anya_def_pre"] - g["a_anya_def_pre"]
    g["epa_off_diff"] = g["h_epa_off_pre"] - g["a_epa_off_pre"]
    g["epa_def_diff"] = g["h_epa_def_pre"] - g["a_epa_def_pre"]
    g["rz_off_diff"] = g["h_rz_off_pre"] - g["a_rz_off_pre"]
    g["rz_def_diff"] = g["h_rz_def_pre"] - g["a_rz_def_pre"]
    g["tov_margin_diff"] = g["h_tov_margin_pre"] - g["a_tov_margin_pre"]
    g["expl_off_diff"] = g["h_expl_off_pre"] - g["a_expl_off_pre"]
    g["expl_def_diff"] = g["h_expl_def_pre"] - g["a_expl_def_pre"]
    g["sos_diff"] = g["h_sos"] - g["a_sos"]
    g["sos_adj_net_epa_diff"] = g["h_sos_adj_net_epa"] - g["a_sos_adj_net_epa"]

    # Injuries: AWAY minus HOME, so positive favors the home team
    g["qb_out_diff"] = g["a_qb_out"] - g["h_qb_out"]
    g["injury_score_diff"] = g["a_inj_score"] - g["h_inj_score"]
    g["wr_lost_diff"] = g["a_wr_lost"] - g["h_wr_lost"]
    for c in INJ_GROUP_COLS:                                  # by position group, Away minus Home
        g[c + "_diff"] = g["a_" + c] - g["h_" + c]
    g["qb_q_diff"] = g["h_qb_q"] - g["a_qb_q"]            # starting QB quality, Home minus Away
    g["qb_pr_diff"] = g["h_qb_pr"] - g["a_qb_pr"]         # starting QB passer rating, Home minus Away

    # ---- weather (recorded at game time; forecasts can be entered in WEATHER_OVERRIDES) ----
    roof = g["roof"].astype(str).str.lower() if "roof" in g.columns else pd.Series("outdoors", index=g.index)
    outdoor = ~roof.isin(["dome", "closed"])
    temp = pd.to_numeric(g["temp"], errors="coerce") if "temp" in g.columns else pd.Series(np.nan, index=g.index)
    wind = pd.to_numeric(g["wind"], errors="coerce") if "wind" in g.columns else pd.Series(np.nan, index=g.index)
    for key, v in WEATHER_OVERRIDES.items():
        m = (g["away_team"] + "@" + g["home_team"] == key) & g["result"].isna()
        if "temp" in v:
            temp = temp.mask(m, v["temp"])
        if "wind" in v:
            wind = wind.mask(m, v["wind"])
    g["wx_wind"] = np.where(outdoor, wind.fillna(0.0), 0.0)
    g["wx_cold"] = np.where(outdoor, (45.0 - temp).clip(lower=0).fillna(0.0), 0.0)
    g["wx_wind_x_qb"] = g["wx_wind"] * g["qb_q_diff"]      # wind hurts the better passing team's edge

    # ---- coaching / scheme, roster continuity ----
    g["aggr_diff"] = g["h_go_pre"] - g["a_go_pre"]                  # 4th-down aggressiveness, Home minus Away
    fade = np.clip(1.0 - (g["week"] - 1) / WIND_FADE_WEEKS, 0.0, 1.0)
    g["cont_early"] = (g["h_cont"] - g["a_cont"]) * fade            # returning production, fades after the first weeks
    g["newqb_early"] = (g["h_newqb"] - g["a_newqb"]) * fade         # new starting QB (negative for home = home has the new QB)
    div = lambda t: t.map(DIVISIONS)
    g["div_game"] = (div(g["home_team"]) == div(g["away_team"])).astype(float)

    g["home_field"] = (g["location"].astype(str).str.lower() != "neutral").astype(int)

    g["vegas_home_prob"] = vegas_home_prob(g)
    g["vegas_spread"] = (pd.to_numeric(g["spread_line"], errors="coerce") if "spread_line" in g.columns
                         else np.nan)              # positive = home favored; NaN becomes 0 in main()
    g = g.merge(compute_common_opponents(g, tg), on="game_id", how="left")
    g = g.merge(elo, on="game_id", how="left")          # Elo (pre-game values)
    g["hfa_diff"] = (g["h_hfa_pre"] - g["a_hfa_pre"]) * g["home_field"]    # 0 at neutral sites
    # ---- style matchups: each side's unit vs the opposing unit, Home minus Away (positive favors home) ----
    g["m_pass"] = (g["h_sty_pass_off_pre"] - g["a_sty_pass_def_pre"]) - (g["a_sty_pass_off_pre"] - g["h_sty_pass_def_pre"])
    g["m_rush"] = (g["h_sty_rush_off_pre"] - g["a_sty_rush_def_pre"]) - (g["a_sty_rush_off_pre"] - g["h_sty_rush_def_pre"])
    g["m_press"] = ((g["h_sty_press_def_pre"] - g["a_sty_press_off_pre"])
                    - (g["a_sty_press_def_pre"] - g["h_sty_press_off_pre"]))      # pass rush vs protection
    g["div_x_elo"] = g["div_game"] * g["elo_diff"]                        # divisional games are usually closer

    # display edges: positive always favors the HOME team
    g["anya_edge"] = g["anya_off_diff"] - g["anya_def_diff"]
    g["epa_edge"] = g["epa_off_diff"] - g["epa_def_diff"]
    g["rz_edge"] = (g["rz_off_diff"] - g["rz_def_diff"]) * 100       # percentage points
    g["expl_edge"] = (g["expl_off_diff"] - g["expl_def_diff"]) * 100 # percentage points
    for k in TOTAL_BASE:                                              # both teams combined
        g["tot_" + k] = g["h_" + k] + g["a_" + k]

    # Off-vs-def unit matchups (0-1 rating scale). The first favors HOME, the second favors AWAY.
    g["rk_home_off_vs_away_def"] = g["h_off_unit"] - g["a_def_unit"]
    g["rk_away_off_vs_home_def"] = g["a_off_unit"] - g["h_def_unit"]
    g["unit_edge"] = (g["rk_home_off_vs_away_def"] - g["rk_away_off_vs_home_def"]) * 100

    g["home_win"] = np.where(g["result"].isna() | (g["result"] == 0), np.nan,
                             (g["result"] > 0).astype(float))
    return g


def model_features():
    feats = _all_model_features()
    if ACTIVE_FEATURE_GROUPS is None:
        return feats
    keep = {c for grp in ACTIVE_FEATURE_GROUPS for c in FEATURE_GROUPS[grp]}
    return [f for f in feats if f in keep]


def _all_model_features():
    sos_cols = ["sos_adj_net_epa_diff"] if USE_SOS_ADJ_IN_MODEL else ["sos_diff"]
    # EPA/play, success rate, ANY/A, explosive plays etc. enter ONLY through the unit rating
    # (rk_* features), so no stat is counted twice.
    return (["rk_home_off_vs_away_def", "rk_away_off_vs_home_def",
             ] + sos_cols +
            ["common_opp_margin_diff", "common_opp_epa_diff", "has_common_opponents",
             "elo_diff", "elo2_diff", "pr_margin", "qb_q_diff", "qb_pr_diff", "qb_out_diff"]
            + (INJ_GROUP_COLS_DIFF if INJURY_BY_POSITION else ["injury_score_diff"]) + ["wr_lost_diff"] + [
             "wx_wind_x_qb", "aggr_diff", "cont_early", "newqb_early", "div_x_elo", "div_game", "m_pass", "m_rush", "m_press",
             "home_field", "hfa_diff"] + (["vegas_spread"] if USE_VEGAS_FEATURE else []))


FEATURE_GROUPS = {
    "Unit rankings": ["rk_home_off_vs_away_def", "rk_away_off_vs_home_def"],
    "SOS": ["sos_diff", "sos_adj_net_epa_diff"],
    "Common Opponents": ["common_opp_margin_diff", "common_opp_epa_diff", "has_common_opponents"],
    "Elo": ["elo_diff"],
    "EPA Elo": ["elo2_diff"],
    "Points ratings": ["pr_margin"],
    "Weather": ["wx_wind_x_qb"],
    "Coaching": ["aggr_diff"],
    "Roster continuity": ["cont_early", "newqb_early"],
    "Divisional": ["div_x_elo", "div_game"],
    "Style matchups": ["m_pass", "m_rush", "m_press"],
    "QB quality": ["qb_q_diff"],
    "QB passer rating": ["qb_pr_diff"],
    "Injuries": ["qb_out_diff", "injury_score_diff"] + INJ_GROUP_COLS_DIFF,
    "Receiving injuries": ["wr_lost_diff"],
    "Home field": ["home_field"],
    "Team HFA": ["hfa_diff"],
    "Vegas line": ["vegas_spread"],
}


# ----------------------------------------------------------------------------
# 8. TRAIN / VALIDATE
# ----------------------------------------------------------------------------
def make_model(C=1.0):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def make_total_model():
    """Ridge regression: projected combined points from both teams' scoring stats."""
    return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=TOTAL_RIDGE_ALPHA))])


def feature_multipliers(feats):
    """Expand GROUP_WEIGHT_MULTIPLIERS into a per-feature dict."""
    mult = {f: 1.0 for f in feats}
    for grp, m in GROUP_WEIGHT_MULTIPLIERS.items():
        if grp not in FEATURE_GROUPS:
            raise ValueError(f"Unknown group '{grp}'. Choose from: {list(FEATURE_GROUPS)}")
        for f in FEATURE_GROUPS[grp]:
            if f in mult:
                mult[f] = float(m)
    return mult


class AdjustedLogit:
    """
    Logistic regression whose learned (standardized) weights are scaled by
    per-feature multipliers. Because scaling changes how extreme the raw scores
    are, a one-variable logistic calibration (slope + intercept) is fit on the
    training data afterwards so the probabilities stay well calibrated. The
    relative emphasis between features is exactly what the multipliers set.
    """
    def __init__(self, feats, multipliers):
        self.feats = list(feats)
        self.mult = np.array([multipliers.get(f, 1.0) for f in self.feats])
        self.base = make_model(MODEL_C)          # the unadjusted, purely learned model

    def fit(self, X, y):
        X = X[self.feats]
        self.base.fit(X, y)
        self.coef_raw = self.base.named_steps["clf"].coef_[0]
        z = self._score(X).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        self.slope, self.bias = cal.coef_[0][0], cal.intercept_[0]
        return self

    def _score(self, X):
        Xs = self.base.named_steps["scaler"].transform(X[self.feats])
        return Xs @ (self.coef_raw * self.mult)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(self.slope * self._score(X) + self.bias)))
        return np.column_stack([1.0 - p, p])

    @property
    def effective_coefs(self):
        """Final standardized weights after multipliers and calibration."""
        return pd.Series(self.coef_raw * self.mult * self.slope, index=self.feats)


# ----------------------------------------------------------------------------
# 8a. WHICH FACTORS ACTUALLY PREDICT WINNERS? (diagnostic on the test seasons)
# ----------------------------------------------------------------------------
def _fit_eval(feats, train, test):
    """Per-game log-loss and per-game 'picked correctly' arrays of a plain logistic model (no multipliers)."""
    y = test["home_win"].values.astype(float)
    if feats:
        p = make_model().fit(train[feats], train["home_win"]).predict_proba(test[feats])[:, 1]
    else:
        p = np.full(len(test), train["home_win"].mean())
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), ((p >= 0.5) == y).astype(float)


def _by_season(feats, done, seasons):
    """Each season is predicted by a model trained on all the OTHER seasons; results are pooled."""
    loss, hit = [], []
    for sn in seasons:
        l, h = _fit_eval(feats, done[done["season"] != sn], done[done["season"] == sn])
        loss.append(l)
        hit.append(h)
    return np.concatenate(loss), np.concatenate(hit)


def feature_ablation(games):
    """
    Factor-accuracy test on EVERY completed season loaded (2016 on), by season-by-season cross-validation:
    each season is predicted by a model trained on the other seasons, and the results are pooled. This gives
    ~1,500+ games instead of one test split, so the comparisons are far less noisy. Features are pre-game
    values only; learned weights only (the multipliers are ignored so this measures information, not tuning).
      Groups : each group ALONE, and the full model WITHOUT that group.
      Stats  : each unit-rating stat ALONE (its offense-vs-defense edge for both teams).
    Lower log-loss = better. 'vs none' is relative to always guessing the home-win rate;
    'noise' is one standard error of that difference, so gaps within about 2x noise are not reliable.
    """
    feats = model_features()
    done = games[games["home_win"].notna() & (games["season"] >= DIAG_START_SEASON)].copy()
    seasons = sorted(done["season"].unique())
    if len(seasons) < 2 or len(done) < 200:
        return None
    n = len(done)
    # ordering inside each season is irrelevant, but keep it stable so arrays line up across runs
    base_loss, base_hit = _by_season([], done, seasons)
    full_loss, full_hit = _by_season(feats, done, seasons)
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        g = [c for c in cols if c in feats]
        if not g:
            continue
        a_loss, a_hit = _by_season(g, done, seasons)
        w_loss, w_hit = _by_season([c for c in feats if c not in g], done, seasons)
        d_alone = a_loss - base_loss
        rows.append({"name": grp, "alone_ll": a_loss.mean(), "alone_acc": float(a_hit.mean()),
                     "alone_vs_none": d_alone.mean(), "noise": d_alone.std(ddof=1) / np.sqrt(n),
                     "without_ll": w_loss.mean(), "without_acc": float(w_hit.mean()),
                     "cost_of_removing": (w_loss - full_loss).mean()})
    stat_rows = []
    for k in RANK_STAT_WEIGHTS:
        need = [f"{h}_ur_{k}_{sd}_pre" for h in "ha" for sd in ("off", "def")]
        if any(c not in done.columns for c in need):
            continue
        so = 1.0 if RANK_DIRECTION[k][0] else -1.0
        sd_ = 1.0 if RANK_DIRECTION[k][1] else -1.0
        done[f"edge_{k}"] = (so * (done[f"h_ur_{k}_off_pre"] - done[f"a_ur_{k}_off_pre"])    # positive = favors home
                             + sd_ * (done[f"h_ur_{k}_def_pre"] - done[f"a_ur_{k}_def_pre"])).fillna(0.0)
        l, h = _by_season([f"edge_{k}"], done, seasons)
        d = l - base_loss
        stat_rows.append({"name": RANK_STAT_NAMES[k], "alone_ll": l.mean(), "alone_acc": float(h.mean()),
                          "alone_vs_none": d.mean(), "noise": d.std(ddof=1) / np.sqrt(n)})
    return {"n_test": n, "seasons": f"{seasons[0]}-{seasons[-1]}", "none_ll": base_loss.mean(),
            "none_acc": float(base_hit.mean()), "full_ll": full_loss.mean(), "full_acc": float(full_hit.mean()),
            "groups": rows, "stats": stat_rows}


COMPARE_CONFIGS = True      # True = run the model set-up comparison (walk-forward) and show it in the report


def compare_configs(games):
    """
    Walk-forward comparison of simpler / differently-regularized set-ups on the SAME games (each season >=
    WF_START_SEASON predicted by a model trained on earlier seasons only), with Vegas shown beside them.
    'vs current' is the paired log-loss difference (negative = better than the shipped model) and 'noise' is
    its standard error, so gaps inside about 2x noise are not real.
    """
    allf = _all_model_features()
    core = ["Elo", "Points ratings", "Unit rankings", "QB quality"]
    active = list(ACTIVE_FEATURE_GROUPS) if ACTIVE_FEATURE_GROUPS else list(FEATURE_GROUPS)
    cfgs = [("Current model (as shipped)", active, True, 1.0)]
    if "EPA Elo" in active:
        cfgs.append(("Same, without the EPA Elo", [g_ for g_ in active if g_ != "EPA Elo"], True, 1.0))
    if "Receiving injuries" in active:
        cfgs.append(("Same, without the receiving-injuries factor", [g_ for g_ in active if g_ != "Receiving injuries"], True, 1.0))
    if "Injuries" in active:           # is the injuries weight too high or too low? (multiplier on its learned weight)
        im = GROUP_WEIGHT_MULTIPLIERS.get("Injuries", 1.0)
        cfgs += [("Injuries at half weight", active, True, 1.0, {"Injuries": im * 0.5}),
                 ("Injuries at 1.5x weight", active, True, 1.0, {"Injuries": im * 1.5})]
    cfgs += [
            ("Same factors, all multipliers 1.0", active, False, 1.0),
            ("Same factors, stronger regularization", active, False, 0.05),
            ("No Injuries", core, False, 1.0),
            ("No Injuries, stronger regularization", core, False, 0.05),
            ("Elo + Points ratings + QB quality", ["Elo", "Points ratings", "QB quality"], False, 0.05),
            ("Points ratings + QB quality", ["Points ratings", "QB quality"], False, 1.0),
            ("Elo + QB quality", ["Elo", "QB quality"], False, 1.0),
            ("Elo only", ["Elo"], False, 1.0)]
    done = games[games["home_win"].notna()]
    seasons = [s_ for s_ in sorted(done["season"].unique()) if s_ >= WF_START_SEASON]
    out, ref = [], None
    for name, groups, use_mult, C, *ov in cfgs:
        feats = [f for f in allf if any(f in FEATURE_GROUPS[g] for g in groups if g in FEATURE_GROUPS)]
        if not feats:
            continue
        ps, ys, vs = [], [], []
        for s_ in seasons:
            tr, te = done[done["season"] < s_], done[done["season"] == s_]
            if len(tr) < WF_MIN_TRAIN_GAMES or te.empty:
                continue
            if use_mult:
                fm_ = feature_multipliers(feats)
                for grp_, mm_ in (ov[0].items() if ov else []):
                    for f_ in FEATURE_GROUPS[grp_]:
                        if f_ in fm_:
                            fm_[f_] = mm_
                m_ = AdjustedLogit(feats, fm_).fit(tr[feats], tr["home_win"])
            else:
                m_ = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000))]
                              ).fit(tr[feats], tr["home_win"])
            ps.append(m_.predict_proba(te[feats])[:, 1])
            ys.append(te["home_win"].values.astype(float))
            vs.append(te["vegas_home_prob"].values if "vegas_home_prob" in te.columns else np.full(len(te), np.nan))
        if not ps:
            continue
        p, y, v = np.clip(np.concatenate(ps), 1e-6, 1 - 1e-6), np.concatenate(ys), np.concatenate(vs)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ref is None:
            ref = ll
        d = ll - ref
        out.append({"name": name, "n_feats": len(feats), "acc": float(((p >= 0.5) == y).mean()), "ll": float(ll.mean()),
                    "vs_cur": float(d.mean()), "noise": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0})
    vm = ~np.isnan(v) if out else None
    veg = None
    if out and vm.any():
        pv = np.clip(v[vm], 1e-6, 1 - 1e-6)
        veg = {"acc": float(((pv >= 0.5) == y[vm]).mean()), "ll": float((-(y[vm] * np.log(pv) + (1 - y[vm]) * np.log(1 - pv))).mean())}
    return {"rows": out, "vegas": veg, "n": int(len(y)) if out else 0}


def print_compare(cp):
    if not cp or not cp["rows"]:
        return
    print("\n" + "=" * 78)
    print(f"MODEL SET-UP COMPARISON (walk-forward, {cp['n']} games, same games for every row)")
    print("=" * 78)
    print(f"  {'set-up':<42s} {'acc':>6s} {'log-loss':>9s} {'vs current':>11s} {'noise':>7s}")
    for r in cp["rows"]:
        print(f"  {r['name']:<42s} {r['acc'] * 100:5.1f}% {r['ll']:9.4f} {r['vs_cur']:+11.4f} {r['noise']:7.4f}")
    if cp["vegas"]:
        print(f"  {'Vegas (no-vig moneyline)':<42s} {cp['vegas']['acc'] * 100:5.1f}% {cp['vegas']['ll']:9.4f}")
    print("  vs current: negative = better than the shipped model; gaps under ~2x noise are not reliable.")


def print_ablation(ab):
    if not ab:
        print("\nFactor accuracy test skipped (not enough completed games).")
        return
    print("\n" + "=" * 78)
    print(f"WHICH FACTORS PREDICT WINNERS?  ({ab['n_test']} games, seasons {ab['seasons']})")
    print("Each season is predicted by a model trained on the other seasons; learned weights only.")
    print("=" * 78)
    print(f"Guess the home-win rate: log-loss {ab['none_ll']:.4f}, accuracy {ab['none_acc']:.3f}")
    print(f"All factors together   : log-loss {ab['full_ll']:.4f}, accuracy {ab['full_acc']:.3f}")
    print("\nEach factor group ALONE, and what the full model loses WITHOUT it")
    print(f"  {'group':<18s} {'alone LL':>9s} {'alone acc':>9s} {'vs none':>9s} {'noise':>7s}   {'lose w/o':>9s}")
    for r in sorted(ab["groups"], key=lambda r: r["alone_vs_none"]):
        print(f"  {r['name']:<18s} {r['alone_ll']:9.4f} {r['alone_acc']:9.3f} {r['alone_vs_none']:+9.4f} "
              f"{r['noise']:7.4f}   {r['cost_of_removing']:+9.4f}")
    print("  (vs none: negative = better than guessing; lose w/o: positive = the group is helping the full model)")
    print("\nEach unit-rating stat ALONE (offense-vs-defense edge for both teams)")
    print(f"  {'stat':<34s} {'alone LL':>9s} {'alone acc':>9s} {'vs none':>9s} {'noise':>7s}")
    for r in sorted(ab["stats"], key=lambda r: r["alone_vs_none"]):
        print(f"  {r['name']:<34s} {r['alone_ll']:9.4f} {r['alone_acc']:9.3f} {r['alone_vs_none']:+9.4f} {r['noise']:7.4f}")
    print("  Differences smaller than about 2x 'noise' are not reliable with this many games.")



def train_and_evaluate(games):
    feats = model_features()
    done = games[games["home_win"].notna()]
    mults = feature_multipliers(feats)

    # ---- walk-forward: each season is predicted by a model trained only on EARLIER seasons ----
    f_nv = [f for f in feats if f != "vegas_spread"] if "vegas_spread" in feats else None
    parts = []
    for s_ in sorted(done["season"].unique()):
        if s_ < WF_START_SEASON:
            continue
        tr, te = done[done["season"] < s_], done[done["season"] == s_]
        if len(tr) < WF_MIN_TRAIN_GAMES or te.empty:
            continue
        m_ = AdjustedLogit(feats, mults).fit(tr[feats], tr["home_win"])
        t_ = make_total_model().fit(tr[TOTAL_FEATS], tr["home_score"] + tr["away_score"])
        part = te.copy()
        part["wf_p"] = m_.predict_proba(te[feats])[:, 1]
        part["wf_p_raw"] = m_.base.predict_proba(te[feats])[:, 1]
        part["wf_total"] = t_.predict(te[TOTAL_FEATS])
        if f_nv:
            mn_ = AdjustedLogit(f_nv, feature_multipliers(f_nv)).fit(tr[f_nv], tr["home_win"])
            part["wf_p_nv"] = mn_.predict_proba(te[f_nv])[:, 1]
        parts.append(part)
    test = pd.concat(parts) if parts else done.iloc[0:0].copy()

    # the model that predicts the upcoming week (all completed games), and one limited to games before
    # PREDICT_SEASON for the fair "record so far this season"
    live_train = done if LIVE_MODEL_USES_ALL_GAMES else done[done["season"] < PREDICT_SEASON]
    model = AdjustedLogit(feats, mults).fit(live_train[feats], live_train["home_win"])
    pre = done[done["season"] < PREDICT_SEASON]
    record_model = AdjustedLogit(feats, mults).fit(pre[feats], pre["home_win"]) if len(pre) >= WF_MIN_TRAIN_GAMES else model
    train = live_train
    report = {"n_train": len(live_train), "n_test": len(test), "record_model": record_model, "acc": None, "ll": None,
              "base_acc": None, "elo_acc": None, "elo_ll": None, "acc_raw": None, "ll_raw": None,
              "feat_mult": mults,
              "group_mult": {g: m for g, m in GROUP_WEIGHT_MULTIPLIERS.items()
                           if m != 1.0 and (ACTIVE_FEATURE_GROUPS is None or g in ACTIVE_FEATURE_GROUPS)}}

    total_model = make_total_model().fit(live_train[TOTAL_FEATS], live_train["home_score"] + live_train["away_score"])
    report.update(total_model=total_model, mae_margin=None, mae_margin_base=None,
                  mae_total=None, mae_total_base=None)

    print("\n" + "=" * 78)
    print("MODEL VALIDATION")
    print("=" * 78)
    if len(test):
        wf_seasons = sorted(test["season"].unique())
        print(f"Walk-forward test: each season {wf_seasons[0]}-{wf_seasons[-1]} is predicted by a model trained only on "
              f"earlier seasons ({len(test)} games pooled)")
        report["wf_first"], report["wf_last"] = int(wf_seasons[0]), int(wf_seasons[-1])
    active = report["group_mult"]
    print("Weight multipliers: " + (", ".join(f"{g} x{m:g}" for g, m in active.items()) or "none"))
    if len(test):
        p = test["wf_p"].values
        acc = accuracy_score(test["home_win"], (p >= 0.5).astype(int))
        ll = log_loss(test["home_win"], p, labels=[0, 1])
        base = test["home_win"].mean()
        p_raw = test["wf_p_raw"].values                          # learned weights, no multipliers
        acc_raw = accuracy_score(test["home_win"], (p_raw >= 0.5).astype(int))
        ll_raw = log_loss(test["home_win"], p_raw, labels=[0, 1])
        print(f"Accuracy : {acc:.3f}   (always-pick-home baseline: {max(base, 1 - base):.3f})")
        print(f"Log-loss : {ll:.4f}  (coin-flip = 0.6931)")
        print(f"Learned weights only (no multipliers): accuracy {acc_raw:.3f}, log-loss {ll_raw:.4f}")
        # standalone Elo (no other features) as a reference point
        pe = test["elo_home_prob"].clip(1e-6, 1 - 1e-6)
        elo_acc = accuracy_score(test["home_win"], (pe >= 0.5).astype(int))
        elo_ll = log_loss(test["home_win"], pe, labels=[0, 1])
        print(f"Elo-only : accuracy {elo_acc:.3f}, log-loss {elo_ll:.4f}")
        if "spread_line" in test.columns:
            tv = test[test["spread_line"].notna() & (test["spread_line"] != 0)]
            if len(tv):
                v_acc = float(((tv["spread_line"] > 0) == (tv["home_win"] == 1)).mean())
                print(f"Vegas favorite (closing spread, a benchmark): accuracy {v_acc:.3f} "
                      f"on {len(tv)} games")
                report["vegas_acc"] = v_acc
        tp = test["vegas_home_prob"] if "vegas_home_prob" in test.columns else None
        if tp is not None and tp.notna().sum() > 50:
            tm_ = test[tp.notna()]
            pv = np.clip(tm_["vegas_home_prob"].values, 1e-6, 1 - 1e-6)
            report["vegas_ll"] = float(log_loss(tm_["home_win"], pv, labels=[0, 1]))
            print(f"Vegas moneyline (no-vig): log-loss {report['vegas_ll']:.4f}  <- the number to beat")
        report.update(acc=acc, ll=ll, base_acc=max(base, 1 - base), elo_acc=elo_acc, elo_ll=elo_ll,
                      acc_raw=acc_raw, ll_raw=ll_raw)
        if "vegas_spread" in feats:
            p_nv = test["wf_p_nv"].values
            report["novegas_acc"] = accuracy_score(test["home_win"], (p_nv >= 0.5).astype(int))
            report["novegas_ll"] = log_loss(test["home_win"], p_nv, labels=[0, 1])
            print(f"WITHOUT the Vegas line: accuracy {report['novegas_acc']:.3f}, log-loss {report['novegas_ll']:.4f}"
                  f"  (the model above uses the spread, so it is partly market-based)")

        # ---- score accuracy: margin from win probability, total from the Ridge model ----
        pm = np.array([MARGIN_SD * NormalDist().inv_cdf(min(max(x, 0.01), 0.99)) for x in p])
        mae_m = float(np.mean(np.abs(pm - test["result"])))
        mae_m0 = float(np.mean(np.abs(test["result"] - done["result"].mean())))
        tot_actual = test["home_score"] + test["away_score"]
        mae_t = float(np.mean(np.abs(test["wf_total"].values - tot_actual)))
        mae_t0 = float(np.mean(np.abs(tot_actual - (done["home_score"] + done["away_score"]).mean())))
        print(f"Score error (avg miss): margin {mae_m:.1f} pts (guessing the average: {mae_m0:.1f}), "
              f"total {mae_t:.1f} pts (guessing the average: {mae_t0:.1f})")
        report.update(mae_margin=mae_m, mae_margin_base=mae_m0, mae_total=mae_t, mae_total_base=mae_t0)
        # season-by-season table (shows whether the model is steadily good or just lucky in one year)
        print("\nWalk-forward results by season")
        print(f"  {'season':>6s} {'games':>6s} {'model acc':>9s} {'model LL':>9s} {'Elo LL':>8s} {'Vegas LL':>9s}")
        wf_rows = []
        for s_, d_ in test.groupby("season"):
            y_ = d_["home_win"].values
            pm_ = np.clip(d_["wf_p"].values, 1e-6, 1 - 1e-6)
            pe_ = np.clip(d_["elo_home_prob"].values, 1e-6, 1 - 1e-6)
            ll_m = float(-np.mean(y_ * np.log(pm_) + (1 - y_) * np.log(1 - pm_)))
            ll_e = float(-np.mean(y_ * np.log(pe_) + (1 - y_) * np.log(1 - pe_)))
            ok_ = d_["vegas_home_prob"].notna().values if "vegas_home_prob" in d_.columns else np.zeros(len(d_), bool)
            ll_v = np.nan
            if ok_.sum() > 20:
                pv_ = np.clip(d_["vegas_home_prob"].values[ok_], 1e-6, 1 - 1e-6)
                yv_ = y_[ok_]
                ll_v = float(-np.mean(yv_ * np.log(pv_) + (1 - yv_) * np.log(1 - pv_)))
            a_ = float(((pm_ >= 0.5) == y_).mean())
            wf_rows.append((int(s_), len(d_), a_, ll_m, ll_e, ll_v))
            print(f"  {int(s_):>6d} {len(d_):>6d} {a_:9.3f} {ll_m:9.4f} {ll_e:8.4f} "
                  + (f"{ll_v:9.4f}" if ll_v == ll_v else f"{'-':>9s}"))
        report["wf_table"] = wf_rows
        pk_ = np.maximum(test["wf_p"].values, 1 - test["wf_p"].values)
        ok_ = ((test["wf_p"].values >= 0.5) == (test["home_win"].values == 1))
        hist_ = []
        for lo_, hi_ in ((0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)):
            m_ = (pk_ >= lo_) & (pk_ < hi_)
            if m_.sum() >= 20:
                hist_.append((lo_, hi_, int(m_.sum()), float(ok_[m_].mean())))
        report["conf_hist"] = hist_
    else:
        print("No completed test games available.")

    # ---- coefficients ----
    coefs = model.effective_coefs
    print("\nStandardized coefficients AFTER multipliers (positive = favors HOME team; |size| = reliance)")
    tbl = pd.DataFrame({"coef": coefs, "odds_ratio_per_1sd": np.exp(coefs)})
    print(tbl.reindex(coefs.abs().sort_values(ascending=False).index).round(4).to_string())
    print("  Note: for *_def_diff features lower = better defense, so a NEGATIVE coef is expected;")
    print("        injury diffs are Away-minus-Home, so a POSITIVE coef is expected.")

    tot = coefs.abs().sum() or 1.0
    by_group = {grp: coefs.reindex([c for c in cols if c in coefs.index]).abs().sum() / tot
                for grp, cols in FEATURE_GROUPS.items() if any(c in coefs.index for c in cols)}
    # "Unit rankings" stays ONE line; its stats are listed underneath it, split by their
    # weights in RANK_STAT_WEIGHTS (approximate: share of the rating's weight, by stat).
    shares = dict(by_group)
    tw = sum(RANK_STAT_WEIGHTS.values())
    unit_parts = {RANK_STAT_NAMES[k]: by_group.get("Unit rankings", 0.0) * w / tw
                  for k, w in RANK_STAT_WEIGHTS.items() if w > 0}
    print("\nShare of total |coefficient| by factor")
    for name, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<34s} {v:6.1%}")
        if name == "Unit rankings":
            for sn, sv in sorted(unit_parts.items(), key=lambda kv: -kv[1]):
                print(f"      - {sn:<28s} {sv:6.1%}")
    report["coefs"], report["shares"], report["unit_parts"] = coefs, shares, unit_parts

    try:
        report["ablation"] = feature_ablation(games)
        print_ablation(report["ablation"])
    except Exception as ex:                          # diagnostic only; never block the predictions
        report["ablation"] = None
        print(f"\n  ! factor accuracy test skipped ({ex})")

    report["compare"] = None
    if COMPARE_CONFIGS:
        try:
            report["compare"] = compare_configs(games)
            print_compare(report["compare"])
        except Exception as ex:
            print(f"\n  ! model set-up comparison skipped ({ex})")

    return model, report


# ----------------------------------------------------------------------------
# 9. UPCOMING WEEK PREDICTIONS
# ----------------------------------------------------------------------------
def season_record(model, games):
    """What this model's picks would have scored so far in PREDICT_SEASON (pre-game features only)."""
    feats = model_features()
    d = games[(games["season"] == PREDICT_SEASON) & games["home_win"].notna()].sort_values(
        ["week", "gameday", "home_team"]).copy()
    if d.empty:
        return None
    d["p_home"] = model.predict_proba(d[feats])[:, 1]
    d["pick_home"] = d["p_home"] >= 0.5
    d["correct"] = d["pick_home"] == (d["home_win"] == 1)
    d["conf"] = np.where(d["pick_home"], d["p_home"], 1 - d["p_home"])
    d["elo_ok"] = (d["elo_home_prob"] >= 0.5) == (d["home_win"] == 1)
    d["home_ok"] = d["home_win"] == 1
    n = len(d)
    vegas = None
    if "spread_line" in d.columns:
        dv = d[d["spread_line"].notna() & (d["spread_line"] != 0)]
        if len(dv):
            vegas = (int(((dv["spread_line"] > 0) == (dv["home_win"] == 1)).sum()), len(dv))
    ties = int(((games["season"] == PREDICT_SEASON) & (games["result"] == 0)).sum())
    wk = d.groupby("week")["correct"].agg(["sum", "size"])
    bands = []
    for lo, hi, label in ((0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "80%+")):
        m = d[(d["conf"] >= lo) & (d["conf"] < hi)]
        if len(m):
            bands.append({"label": label, "w": int(m["correct"].sum()), "n": len(m)})
    rows = [{"week": int(r.week), "away": r.away_team, "home": r.home_team,
             "pick": r.home_team if r.pick_home else r.away_team, "conf": float(r.conf),
             "score": f"{int(r.away_score)}-{int(r.home_score)}", "ok": bool(r.correct)}
            for r in d.itertuples()]
    return {"n": n, "w": int(d["correct"].sum()), "ties": ties,
            "elo_w": int(d["elo_ok"].sum()), "home_w": int(d["home_ok"].sum()), "vegas": vegas,
            "weeks": [(int(k), int(v["sum"]), int(v["size"])) for k, v in wk.iterrows()],
            "bands": bands, "rows": rows}


def print_season_record(rec):
    print("\n" + "=" * 78)
    print(f"MODEL RECORD SO FAR IN {PREDICT_SEASON}")
    print("=" * 78)
    if not rec:
        print("No completed games yet.")
        return
    n, w = rec["n"], rec["w"]
    print(f"Model picks : {w}-{n - w}  ({100 * w / n:.1f}%)" + (f"   [{rec['ties']} tie(s) not counted]" if rec["ties"] else ""))
    print(f"Elo only    : {rec['elo_w']}-{n - rec['elo_w']}  ({100 * rec['elo_w'] / n:.1f}%)")
    print(f"Always home : {rec['home_w']}-{n - rec['home_w']}  ({100 * rec['home_w'] / n:.1f}%)")
    if rec.get("vegas"):
        vw, vn = rec["vegas"]
        print(f"Vegas favorite (closing spread): {vw}-{vn - vw}  ({100 * vw / vn:.1f}%)   <- the benchmark to beat")
    print("By week     : " + "  ".join(f"W{k} {a}-{b - a}" for k, a, b in rec["weeks"]))
    print("By confidence: " + "  ".join(f"{b['label']} {b['w']}-{b['n'] - b['w']}" for b in rec["bands"]))
    if REFIT_ON_ALL_COMPLETED:
        print("  NOTE: REFIT_ON_ALL_COMPLETED is on, so these games were in the model's training data (not a fair record).")



STYLE_NOTE_Z = 1.2     # a style matchup is called out when the unit gap is at least this many league standard deviations


def style_notes(r, lg):
    """Notable pass / rush / pass-rush matchups for one game (display only; not used by the model)."""
    def z(col):
        m, sd = lg[col]
        return (getattr(r, col) - m) / sd if sd > 0 else 0.0
    out = []
    for off_t, def_t, o, d in ((r.away_team, r.home_team, "a", "h"), (r.home_team, r.away_team, "h", "a")):
        for what, k, lab in (("pass", "pass", "passing game"), ("rush", "rush", "running game")):
            adv = z(f"{o}_sty_{k}_off_pre") - z(f"{d}_sty_{k}_def_pre")          # offense EPA up, defense EPA allowed down
            out.append((adv, off_t, def_t, lab, "off"))
        adv = z(f"{d}_sty_press_def_pre") - z(f"{o}_sty_press_off_pre")           # pressure generated vs allowed
        out.append((adv, off_t, def_t, "pass protection", "press"))
    notes = []
    for adv, off_t, def_t, lab, kind in sorted(out, key=lambda t: -abs(t[0])):
        if abs(adv) < STYLE_NOTE_Z:
            continue
        if kind == "press":
            txt = (f"{def_t} pass rush vs {off_t} protection: edge to {def_t}" if adv > 0
                   else f"{off_t} protection vs {def_t} pass rush: edge to {off_t}")
        else:
            txt = (f"{off_t} {lab} vs {def_t} defense: edge to {off_t}" if adv > 0
                   else f"{def_t} defense vs {off_t} {lab}: edge to {def_t}")
        notes.append((txt, abs(adv)))
    return notes[:3]


def group_contributions(model, X):
    """Per-game log-odds each factor GROUP adds for the HOME team, relative to an average game (positive = favors home)."""
    Xs = model.base.named_steps["scaler"].transform(X[model.feats])
    c = pd.DataFrame(Xs * (model.coef_raw * model.mult * model.slope), columns=model.feats, index=X.index)
    out = {}
    for grp, cols in FEATURE_GROUPS.items():
        cc = [x for x in cols if x in c.columns]
        if cc:
            out[grp] = c[cc].sum(axis=1)
    return pd.DataFrame(out)


def confidence_words(p):
    if p >= 0.72:
        return "Very confident"
    if p >= 0.65:
        return "Confident"
    if p >= 0.58:
        return "Moderate lean"
    if p >= 0.54:
        return "Slight lean"
    return "Toss-up"


def build_why(r, gc, hist):
    """Plain-language reasons for the pick and concerns about it. r = one itertuples row; gc = {group: home log-odds}."""
    home_pick = r.p_home >= 0.5
    P, O = (r.home_team, r.away_team) if home_pick else (r.away_team, r.home_team)
    sg = 1.0 if home_pick else -1.0
    pp = max(r.p_home, 1 - r.p_home)
    v = {g: sg * float(x) for g, x in gc.items()}                       # positive = supports the pick
    pre = "h" if home_pick else "a"
    opp = "a" if home_pick else "h"
    g = lambda side, name: getattr(r, f"{side}_{name}")
    rk = lambda x: f"#{int(round(x))}"
    reasons, concerns = [], []

    def unit_text(tm_off, so, tm_def, sd):
        """sentence about so (offense) vs sd (defense): returns (edge, text)"""
        o_u, o_r = g(so, "off_unit"), g(so, "off_rank")
        d_u, d_r = g(sd, "def_unit"), g(sd, "def_rank")
        return o_u - d_u, f"{tm_off}'s offense ({rk(o_r)} in the league) against {tm_def}'s defense ({rk(d_r)})"

    # ---- reasons ----
    rating_v = v.get("Elo", 0.0) + v.get("EPA Elo", 0.0)
    if rating_v >= 0.08:
        t = f"{P} is the stronger team on overall rating (Elo {g(pre, 'elo_pre'):.0f} vs {g(opp, 'elo_pre'):.0f})"
        if v.get("EPA Elo", 0.0) >= 0.04:
            t += ", and its efficiency-based rating (EPA Elo) agrees"
        reasons.append((rating_v, t + "."))
    if v.get("Points ratings", 0.0) >= 0.08:
        reasons.append((v["Points ratings"], f"{P} has been beating opponents by more points than {O} once schedule strength is accounted for."))
    if v.get("Unit rankings", 0.0) >= 0.08:
        e1, t1 = unit_text(P, pre, O, opp)
        e2, t2 = unit_text(O, opp, P, pre)
        if e1 - e2 >= 0:
            reasons.append((v["Unit rankings"], f"Favorable matchup: {t1}."))
        else:
            reasons.append((v["Unit rankings"], f"Matchup edge: {P}'s defense ({rk(g(pre, 'def_rank'))}) holds up well against {O}'s offense ({rk(g(opp, 'off_rank'))})."))
    if v.get("QB quality", 0.0) >= 0.08:
        reasons.append((v["QB quality"], f"{P} has the quarterback edge (QB play has been better per dropback)."))
    if v.get("Injuries", 0.0) >= 0.08:
        reasons.append((v["Injuries"], f"{O} is dealing with more key injuries than {P}."))
    if int(g(opp, "qb_out")):
        reasons.append((0.5, f"{O}'s usual starting quarterback is listed out."))
    def key_out(team):
        k = [z for z in INJ_KEY_NOW.get(team, []) if z[2] != "QB"][:3]       # a missing QB has its own line
        return ", ".join(f"{n_} ({p_}, {st_})" for _, n_, p_, st_ in k)
    ko_o, ko_p = key_out(O), key_out(P)
    lo_, lp_ = float(g(opp, "wr_lost") or 0), float(g(pre, "wr_lost") or 0)
    if lo_ >= 0.15:
        reasons.append((0.3 + lo_, f"{O} is missing receivers who had about {lo_ * 100:.0f}% of their targets."))
    if ko_o:
        reasons.append((0.3 + sum(w_ for w_, *_ in INJ_KEY_NOW.get(O, [])[:3]) / 10, f"{O} is missing key players: {ko_o}."))
    reasons.sort(key=lambda t: -t[0])
    reasons = [t for _, t in reasons[:4]]
    if not reasons:
        reasons = [f"No single factor stands out; {P} gets a small edge from several factors added together."]

    # ---- concerns ----
    cs = []
    if pp < 0.57:
        cs.append((1.0, f"Very close call: {P} is only {pp * 100:.0f}% to win, barely better than a coin flip."))
    if v.get("Unit rankings", 0.0) <= -0.08:
        e2, t2 = unit_text(O, opp, P, pre)
        cs.append((-v["Unit rankings"], f"Matchup risk: {t2} leans toward {O}."))
    if v.get("Elo", 0.0) + v.get("EPA Elo", 0.0) <= -0.08:
        cs.append((-(v.get("Elo", 0.0) + v.get("EPA Elo", 0.0)), f"{O} actually rates higher overall (Elo {g(opp, 'elo_pre'):.0f} vs {g(pre, 'elo_pre'):.0f}); the pick leans on other factors."))
    if v.get("Points ratings", 0.0) <= -0.08:
        cs.append((-v["Points ratings"], f"{O} has the better points-based rating."))
    if v.get("QB quality", 0.0) <= -0.08:
        cs.append((-v["QB quality"], f"{O} has the quarterback edge."))
    if v.get("Injuries", 0.0) <= -0.08:
        cs.append((-v["Injuries"], f"{P} has more key injuries than {O}."))
    if int(g(pre, "qb_out")):
        cs.append((1.0, f"{P}'s usual starting quarterback is listed out, so the rating may overstate them."))
    if lp_ >= 0.15:
        cs.append((0.8 + lp_, f"{P} is missing receivers who had about {lp_ * 100:.0f}% of their targets."))
    if ko_p:
        cs.append((0.8 + sum(w_ for w_, *_ in INJ_KEY_NOW.get(P, [])[:3]) / 10, f"{P} is missing key players: {ko_p}."))
    if g(pre, "def_rank") >= 22:
        cs.append((0.6, f"{P}'s defense ranks {rk(g(pre, 'def_rank'))} in the league; the pick leans on their offense and overall rating."))
    if g(pre, "off_rank") >= 22:
        cs.append((0.6, f"{P}'s offense ranks {rk(g(pre, 'off_rank'))} in the league; the pick leans on their defense and overall rating."))
    vp = getattr(r, "vegas_home_prob", np.nan)
    if vp is not None and not pd.isna(vp):
        mk = vp if home_pick else 1 - vp
        if mk < 0.5:
            cs.append((0.9, f"Vegas disagrees on the winner: the betting market has {O} favored ({(1 - mk) * 100:.0f}% for {O}, while the model gives {P} {pp * 100:.0f}%)."))
        elif (pp - mk) * 100 <= -MARKET_FLAG_PTS:
            cs.append((0.7, f"Vegas is more confident in {P} ({mk * 100:.0f}% vs the model's {pp * 100:.0f}%)."))
        elif (pp - mk) * 100 >= MARKET_FLAG_PTS:
            cs.append((0.7, f"The model likes {P} much more than the betting market does ({pp * 100:.0f}% vs {mk * 100:.0f}%); the market may know something the model does not."))
    wind = getattr(r, "wx_wind", 0.0)
    if wind is not None and not pd.isna(wind) and wind >= 15:
        cs.append((0.4, f"Windy conditions ({wind:.0f} mph) add randomness; the pick does not adjust for weather."))
    if int(r.week) <= 4:
        cs.append((0.3, "Early in the season: ratings still lean partly on last year, so they are less reliable."))
    cs.sort(key=lambda t: -t[0])
    concerns = [t for _, t in cs[:5]]
    if not concerns:
        concerns = ["Nothing stands out: the factors agree with the pick and the betting market is on the same side."]

    hist_txt = ""
    for lo, hi, n, acc in (hist or []):
        if lo <= pp < hi:
            hist_txt = f"Picks in this confidence range ({lo * 100:.0f}-{min(hi, 1) * 100:.0f}%) have been right {acc * 100:.0f}% of the time in past seasons ({n} games)."
    names = {"Elo": "Elo", "EPA Elo": "EPA Elo", "Points ratings": "Points ratings", "Unit rankings": "Matchups",
             "QB quality": "QB", "Injuries": "Injuries", "Receiving injuries": "Receivers out"}
    shifts = [(names.get(g_, g_), 100.0 * pp * (1 - pp) * x) for g_, x in v.items()]       # approx. points of win probability
    shifts.sort(key=lambda z: -abs(z[1]))
    return dict(word=confidence_words(pp), pick=P, other=O, pp=pp, reasons=reasons, concerns=concerns, hist=hist_txt, shifts=shifts)


def fetch_forecasts(wk):
    """
    {game_id: {"temp": F, "wind": mph, "gust": mph, "rain": chance %, "dome": bool}} at kickoff for each unplayed game.
    Open-Meteo hourly forecast at the home stadium. Domes / closed roofs are marked and not fetched. Anything that fails
    simply has no entry (the model then treats that game as calm and mild).
    """
    import json
    from datetime import timedelta, timezone
    out = {}
    if not WEATHER_FORECAST:
        return out
    for r in wk.itertuples():
        if pd.notna(r.result):
            continue
        roof = str(getattr(r, "roof", "") or "").lower()
        if roof in ("dome", "closed"):
            out[r.game_id] = {"dome": True}
            continue
        lat_lon = TEAM_COORDS.get(r.home_team)
        if lat_lon is None or str(getattr(r, "location", "Home")) == "Neutral":
            continue
        try:
            day = pd.to_datetime(r.gameday)
            hh, mm = (str(getattr(r, "gametime", "13:00") or "13:00").split(":") + ["0"])[:2]
            ko_et = day.to_pydatetime().replace(hour=int(hh), minute=int(mm))
            # gametime is Eastern: UTC-5 after the clocks go back (early Nov), UTC-4 before (approx. DST by date)
            is_dst = pd.Timestamp(ko_et).month in (3, 4, 5, 6, 7, 8, 9, 10) and not (pd.Timestamp(ko_et).month == 3 and pd.Timestamp(ko_et).day < 9)
            ko_utc = ko_et + timedelta(hours=4 if is_dst else 5)
            if (ko_utc - datetime.utcnow()).days > 14:
                continue
            url = ("https://api.open-meteo.com/v1/forecast?latitude={:.3f}&longitude={:.3f}"
                   "&hourly=temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability"
                   "&temperature_unit=fahrenheit&wind_speed_unit=mph&forecast_days=16&timezone=UTC").format(*lat_lon)
            with urllib.request.urlopen(url, timeout=20) as resp:
                d = json.loads(resp.read().decode("utf-8"))["hourly"]
            want = ko_utc.strftime("%Y-%m-%dT%H:00")
            i = d["time"].index(want)
            hrs = range(i, min(i + 3, len(d["time"])))                     # about the length of the game
            g_ = lambda k: [d[k][j] for j in hrs if d.get(k) and d[k][j] is not None]
            out[r.game_id] = {"temp": float(np.mean(g_("temperature_2m"))), "wind": float(np.mean(g_("wind_speed_10m"))),
                              "gust": float(np.max(g_("wind_gusts_10m"))), "rain": float(np.max(g_("precipitation_probability"))),
                              "dome": False}
        except Exception as ex:
            print(f"  ! no forecast for {r.away_team} @ {r.home_team} ({ex})")
    got = sum(1 for v in out.values() if not v.get("dome") and "temp" in v)
    print(f"Weather forecasts: {got} outdoor game(s) found" + ("" if got else " (none; treating games as calm and mild)"))
    return out


def predict_upcoming(model, games, report):
    feats = model_features()
    pending = games[(games["season"] == PREDICT_SEASON) & games["result"].isna()]
    if pending.empty:
        print(f"\nNo unplayed {PREDICT_SEASON} regular-season games found.")
        return None
    week = pick_week(games)
    if INJ_WEEK_NOW and INJ_WEEK_NOW != week:      # injury data is for a different week than the one being predicted
        print(f"  ! No injury reports for week {week} yet (latest data is week {INJ_WEEK_NOW}); picks assume healthy rosters. "
              f"Teams on short weeks report first; rerun after their reports post.")
        INJ_KEY_NOW.clear()
        INJ_IMPACT_NOW.clear()
    # the WHOLE week stays on screen: games already played keep their pre-game pick next to the final score
    wk = games[(games["season"] == PREDICT_SEASON) & (games["week"] == week)].sort_values(
        ["gameday", "home_team"]).copy()

    wxf = fetch_forecasts(wk)
    for gid_, v_ in wxf.items():                       # forecast replaces "calm and mild" (manual entries in WEATHER_OVERRIDES win)
        m_ = wk["game_id"] == gid_
        key_ = (wk.loc[m_, "away_team"] + "@" + wk.loc[m_, "home_team"]).iloc[0]
        if v_.get("dome") or key_ in WEATHER_OVERRIDES or "temp" not in v_:
            continue
        wk.loc[m_, "wx_wind"] = v_["wind"]
        wk.loc[m_, "wx_cold"] = max(45.0 - v_["temp"], 0.0)
        wk.loc[m_, "wx_wind_x_qb"] = v_["wind"] * wk.loc[m_, "qb_q_diff"]
    wk["p_home"] = model.predict_proba(wk[feats])[:, 1]
    _sty = {}
    _ssn = games[games["season"] == PREDICT_SEASON]
    for _k in ("pass_off", "pass_def", "rush_off", "rush_def", "press_off", "press_def"):
        for _c in (f"h_sty_{_k}_pre", f"a_sty_{_k}_pre"):
            _v = pd.concat([_ssn[f"h_sty_{_k}_pre"], _ssn[f"a_sty_{_k}_pre"]])
            _sty[_c] = (float(_v.mean()), float(_v.std()))
    wk["winner"] = np.where(wk["p_home"] >= 0.5, wk["home_team"], wk["away_team"])
    try:
        _gc = group_contributions(model, wk[feats]) if hasattr(model, "coef_raw") else None
    except Exception as ex:
        print(f"  ! per-game explanations unavailable ({ex})")
        _gc = None

    # ---- predicted score: projected total split by the margin implied by the win probability ----
    wk["pred_margin"] = [MARGIN_SD * NormalDist().inv_cdf(min(max(x, 0.01), 0.99)) for x in wk["p_home"]]
    wk["pred_total"] = np.clip(report["total_model"].predict(wk[TOTAL_FEATS]), 30, 65)
    hs = np.rint(np.maximum((wk["pred_total"] + wk["pred_margin"]) / 2, 0)).astype(int)
    as_ = np.rint(np.maximum((wk["pred_total"] - wk["pred_margin"]) / 2, 0)).astype(int)
    tie = hs == as_                                    # never show a tie for a predicted winner
    hs = np.where(tie & (wk["p_home"] >= 0.5), hs + 1, hs)
    as_ = np.where(tie & (wk["p_home"] < 0.5), as_ + 1, as_)
    wk["score_home"], wk["score_away"] = hs, as_
    out = pd.DataFrame({
        "Matchup": wk["away_team"] + " @ " + wk["home_team"],
        "Predicted Winner": wk["winner"],
        "Home Win Prob %": (wk["p_home"] * 100).round(1),
        "Predicted Score": (wk["away_team"] + " " + wk["score_away"].astype(str) + " - "
                            + wk["score_home"].astype(str) + " " + wk["home_team"]),
        "ANY/A Edge": wk["anya_edge"].map("{:+.2f}".format),
        "EPA Edge": wk["epa_edge"].map("{:+.3f}".format),
        "RZ Edge": wk["rz_edge"].map("{:+.1f}".format),
        "TO Edge": wk["tov_margin_diff"].map("{:+.2f}".format),
        "Expl Edge": wk["expl_edge"].map("{:+.1f}".format),
        "Unit Edge": wk["unit_edge"].map("{:+.0f}".format),
        "SOS Edge": wk["sos_adj_net_epa_diff"].map("{:+.3f}".format),
        "Common Opp Edge": wk["common_opp_margin_diff"].map("{:+.1f}".format)
                           + " (" + wk["n_common_opps"].astype(int).astype(str) + ")",
        "HFA Edge": wk["hfa_diff"].map("{:+.1f}".format),
        "Inj Edge": wk["injury_score_diff"].map("{:+.1f}".format),
        "QB Out (A/H)": wk["a_qb_out"].astype(int).astype(str) + "/" + wk["h_qb_out"].astype(int).astype(str),
    })
    # result column for games already played (the pick above used pre-game information only)
    def _res(r):
        if pd.isna(r.result):
            return ""
        if r.result == 0:
            return "tie"
        return f"{int(r.away_score)}-{int(r.home_score)} " + ("RIGHT" if (r.p_home >= 0.5) == (r.result > 0) else "WRONG")
    out["Final"] = [_res(r) for r in wk.itertuples()]
    if not (out["Final"] != "").any():
        out = out.drop(columns=["Final"])

    print("\n" + "=" * 78)
    n_final = int(wk["result"].notna().sum())
    print(f"{PREDICT_SEASON} WEEK {week} PREDICTIONS  ({len(out)} games, {n_final} already final)")
    print("=" * 78)
    print(out.to_string(index=False))

    # ---- separate Elo area: raw pre-game Elo numbers only ----
    elo_tbl = pd.DataFrame({
        "Away Team": wk["away_team"].values,
        "Away Elo": wk["a_elo_pre"].round(0).astype(int).values,
        "Home Team": wk["home_team"].values,
        "Home Elo": wk["h_elo_pre"].round(0).astype(int).values,
    })
    print("\n" + "=" * 78)
    print(f"ELO RATINGS FOR WEEK {week} MATCHUPS")
    print("=" * 78)
    print(elo_tbl.to_string(index=False))

    unit_tbl = pd.DataFrame({
        "Away Team": wk["away_team"].values,
        "Away Off #": wk["a_off_rank"].astype(int).values, "Away Def #": wk["a_def_rank"].astype(int).values,
        "Home Team": wk["home_team"].values,
        "Home Off #": wk["h_off_rank"].astype(int).values, "Home Def #": wk["h_def_rank"].astype(int).values})
    print("\n" + "=" * 78)
    print(f"OFFENSE / DEFENSE UNIT RANKINGS FOR WEEK {week} MATCHUPS (1 = best in the league)")
    print("=" * 78)
    print(unit_tbl.to_string(index=False))

    print("\nEdges are Home-minus-Away; positive favors the HOME team.")
    print("  ANY/A Edge  = (home off - away off) + (away def allowed - home def allowed)")
    print("  EPA Edge    = same construction using EPA/play")
    print("  RZ Edge     = red-zone TD% edge in points: (home off - away off) + (away def allowed - home def allowed)")
    print("  Predicted Score = projected total points (from both teams' scoring stats) split by the win-probability margin")
    print("  Unit Edge   = (home offense rating vs away defense) minus (away offense vs home defense), 0-100 scale")
    print("  (ANY/A, EPA and Expl edges are for display; those stats feed the model through the unit rating.)")
    print("  TO Edge     = turnover margin per game (last 8), home minus away")
    print("  Expl Edge   = explosive-play rate edge in points: (home off - away off) + (away def allowed - home def allowed)")
    print("  SOS Edge    = SOS-adjusted net EPA difference (net EPA + opponent strength)")
    print("  Common Opp  = margin diff vs common opponents; (n) = number of common opponents")
    print("  HFA Edge    = home team's home-field advantage minus away team's (points)")
    print("  Inj Edge    = Away injury impact - Home injury impact; QB Out = starting QB flag")
    no_injuries = bool((wk["h_inj_score"].eq(0) & wk["a_inj_score"].eq(0)).all())
    if no_injuries:
        print("  ! No Out/Doubtful injuries found for this week (report may not be published yet).")

    rows = [dict(away=r.away_team, home=r.home_team, winner=r.winner, p_home=r.p_home,
                 score_a=int(r.score_away), score_h=int(r.score_home),
                 n_common=int(r.n_common_opps),
                 a_elo=r.a_elo_pre, h_elo=r.h_elo_pre,
                 a_anya_off=r.a_anya_off_pre, h_anya_off=r.h_anya_off_pre,
                 a_anya_def=r.a_anya_def_pre, h_anya_def=r.h_anya_def_pre,
                 a_epa_off=r.a_epa_off_pre, h_epa_off=r.h_epa_off_pre,
                 a_epa_def=r.a_epa_def_pre, h_epa_def=r.h_epa_def_pre,
                 a_net=r.a_net_epa_pre, h_net=r.h_net_epa_pre,
                 a_sos=r.a_sos, h_sos=r.h_sos,
                 a_sosadj=r.a_sos_adj_net_epa, h_sosadj=r.h_sos_adj_net_epa,
                 a_hfa=r.a_hfa_pre, h_hfa=r.h_hfa_pre,
                 a_inj=r.a_inj_score, h_inj=r.h_inj_score,
                 a_qb=int(r.a_qb_out), h_qb=int(r.h_qb_out),
                 a_off_unit=r.a_off_unit, h_off_unit=r.h_off_unit,
                 a_def_unit=r.a_def_unit, h_def_unit=r.h_def_unit,
                 a_off_rank=r.a_off_rank, h_off_rank=r.h_off_rank,
                 a_def_rank=r.a_def_rank, h_def_rank=r.h_def_rank,
                 a_succ_off=100 * r.a_succ_off_pre, h_succ_off=100 * r.h_succ_off_pre,
                 a_succ_def=100 * r.a_succ_def_pre, h_succ_def=100 * r.h_succ_def_pre,
                 a_tov=r.a_tov_margin_pre, h_tov=r.h_tov_margin_pre,
                 a_expl_off=100 * r.a_expl_off_pre, h_expl_off=100 * r.h_expl_off_pre,
                 a_expl_def=100 * r.a_expl_def_pre, h_expl_def=100 * r.h_expl_def_pre,
                 a_rz_off=100 * r.a_rz_off_pre, h_rz_off=100 * r.h_rz_off_pre,
                 a_rz_def=100 * r.a_rz_def_pre, h_rz_def=100 * r.h_rz_def_pre,
                 a_cm=r.a_common_margin, h_cm=r.h_common_margin,
                 a_ce=r.a_common_epa, h_ce=r.h_common_epa,
                 why=(build_why(r, _gc.loc[r.Index].to_dict(), report.get("conf_hist")) if _gc is not None else None),
                 wx=wxf.get(r.game_id) or ({"temp": float(r.temp), "wind": float(r.wind), "dome": False, "recorded": True}
                                           if (pd.notna(getattr(r, "temp", np.nan)) and pd.notna(getattr(r, "wind", np.nan))
                                               and str(getattr(r, "roof", "")).lower() not in ("dome", "closed")) else None),
                 roof_=str(getattr(r, "roof", "") or "").lower(),
                 played=bool(pd.notna(r.result)), final_a=r.away_score, final_h=r.home_score,
                 pred_margin=float(r.pred_margin),
                 v_ml_home=getattr(r, "home_moneyline", np.nan), v_ml_away=getattr(r, "away_moneyline", np.nan),
                 v_spread=getattr(r, "spread_line", np.nan), v_total=getattr(r, "total_line", np.nan),
                 v_prob=getattr(r, "vegas_home_prob", np.nan), style=style_notes(r, _sty))
            for r in wk.itertuples()]
    return {"week": week, "rows": rows, "no_injuries": no_injuries}


# ----------------------------------------------------------------------------
# SEASON TEAM STATS & RANKINGS (descriptive; for display, not model inputs)
# ----------------------------------------------------------------------------
# (section, label, column, higher_is_better, format)
STAT_SPECS = [
    ("Offense", "Points / game", "off_ppg", True, "{:.1f}"),
    ("Offense", "Total yards / game", "off_ypg", True, "{:.0f}"),
    ("Offense", "Pass yards / game (net of sacks)", "off_pass_ypg", True, "{:.0f}"),
    ("Offense", "Rush yards / game", "off_rush_ypg", True, "{:.0f}"),
    ("Offense", "EPA / play", "off_epa", True, "{:+.3f}"),
    ("Offense", "Success rate", "off_succ", True, "{:.1f}%"),
    ("Offense", "Rush EPA / play", "off_rush_epa", True, "{:+.3f}"),
    ("Offense", "Pressure allowed (sacks + QB hits / dropback)", "off_press", False, "{:.1f}%"),
    ("Offense", "Scoring-drive rate (TD or FG)", "off_drive", True, "{:.1f}%"),
    ("Offense", "ANY/A", "off_anya", True, "{:.2f}"),
    ("Offense", "3rd-down conversion %", "off_3d", True, "{:.1f}%"),
    ("Offense", "Red-zone TD %", "off_rz", True, "{:.1f}%"),
    ("Offense", "Explosive plays / game (10+ run, 20+ pass)", "off_expl", True, "{:.1f}"),
    ("Offense", "Avg drive start (own yard line)", "off_fp", True, "{:.1f}"),
    ("Offense", "Turnovers (giveaways)", "off_tov", False, "{:.0f}"),
    ("Offense", "Sacks taken", "off_sacks", False, "{:.0f}"),
    ("Defense", "Points allowed / game", "def_ppg", False, "{:.1f}"),
    ("Defense", "Yards allowed / game", "def_ypg", False, "{:.0f}"),
    ("Defense", "Pass yards allowed / game", "def_pass_ypg", False, "{:.0f}"),
    ("Defense", "Rush yards allowed / game", "def_rush_ypg", False, "{:.0f}"),
    ("Defense", "EPA / play allowed", "def_epa", False, "{:+.3f}"),
    ("Defense", "Success rate allowed", "def_succ", False, "{:.1f}%"),
    ("Defense", "Rush EPA / play allowed", "def_rush_epa", False, "{:+.3f}"),
    ("Defense", "Pressure rate generated", "def_press", True, "{:.1f}%"),
    ("Defense", "Scoring-drive rate allowed", "def_drive", False, "{:.1f}%"),
    ("Defense", "ANY/A allowed", "def_anya", False, "{:.2f}"),
    ("Defense", "3rd-down % allowed", "def_3d", False, "{:.1f}%"),
    ("Defense", "Red-zone TD % allowed", "def_rz", False, "{:.1f}%"),
    ("Defense", "Explosive plays allowed / game", "def_expl", False, "{:.1f}"),
    ("Defense", "Opp avg drive start (their yard line)", "def_fp", False, "{:.1f}"),
    ("Defense", "Takeaways", "def_tov", True, "{:.0f}"),
    ("Defense", "Sacks", "def_sacks", True, "{:.0f}"),
    ("Overall", "Point differential / game", "pd_pg", True, "{:+.1f}"),
    ("Overall", "Turnover margin", "tov_margin", True, "{:+.0f}"),
    ("Overall", "Net EPA / play", "net_epa", True, "{:+.3f}"),
]


# (label, offense column, defense column, format): an offense is shown against the
# opposing defense's matching stat. Rank 1 = best in the league on each side.
MATCHUP_PAIRS = [
    ("Points / game \u2605", "off_ppg", "def_ppg", "{:.1f}"),
    ("Total yards / game \u2605", "off_ypg", "def_ypg", "{:.0f}"),
    ("Pass yards / game", "off_pass_ypg", "def_pass_ypg", "{:.0f}"),
    ("Rush yards / game", "off_rush_ypg", "def_rush_ypg", "{:.0f}"),
    ("EPA / play \u2605", "off_epa", "def_epa", "{:+.3f}"),
    ("Success rate \u2605", "off_succ", "def_succ", "{:.1f}%"),
    ("Rush EPA / play \u2605", "off_rush_epa", "def_rush_epa", "{:+.3f}"),
    ("Pressure: allowed vs generated \u2605", "off_press", "def_press", "{:.1f}%"),
    ("Scoring-drive rate \u2605", "off_drive", "def_drive", "{:.1f}%"),
    ("ANY/A \u2605", "off_anya", "def_anya", "{:.2f}"),
    ("3rd-down % \u2605", "off_3d", "def_3d", "{:.1f}%"),
    ("Red-zone TD % \u2605", "off_rz", "def_rz", "{:.1f}%"),
    ("Explosive plays / game \u2605", "off_expl", "def_expl", "{:.1f}"),
    ("Avg starting field position (yd line)", "off_fp", "def_fp", "{:.1f}"),
    ("Giveaways vs takeaways \u2605", "off_tov", "def_tov", "{:.0f}"),
    ("Sacks taken vs sacks", "off_sacks", "def_sacks", "{:.0f}"),
]


def build_season_team_stats(sched, pbp, season):
    """
    Season-to-date team stats (completed games only) with league rankings.
    Every *_rk column is a rank where 1 = best in the league for that stat
    (lower is better for allowed yards/points, giveaways, sacks taken, etc.).
    """
    played = sched[(sched["season"] == season) & sched["result"].notna()]
    if played.empty:
        return None
    ids = set(played["game_id"])

    g = pd.concat([
        pd.DataFrame({"team": played["home_team"], "pf": played["home_score"], "pa": played["away_score"], "home": 1}),
        pd.DataFrame({"team": played["away_team"], "pf": played["away_score"], "pa": played["home_score"], "home": 0})])
    g["w"], g["l"], g["t"] = (g["pf"] > g["pa"]).astype(int), (g["pf"] < g["pa"]).astype(int), (g["pf"] == g["pa"]).astype(int)
    st = g.groupby("team").agg(games=("pf", "size"), pf=("pf", "sum"), pa=("pa", "sum"),
                               w=("w", "sum"), l=("l", "sum"), t=("t", "sum"))
    st["record"] = st["w"].astype(str) + "-" + st["l"].astype(str) + np.where(st["t"] > 0, "-" + st["t"].astype(str), "")
    for flag, name in ((1, "home_record"), (0, "away_record")):          # home / road splits
        sub = g[g["home"] == flag].groupby("team")[["w", "l", "t"]].sum()
        st[name] = [f"{int(sub.loc[t, 'w'])}-{int(sub.loc[t, 'l'])}" + (f"-{int(sub.loc[t, 't'])}" if sub.loc[t, "t"] else "")
                    if t in sub.index else "0-0" for t in st.index]
    n = st["games"]
    st["off_ppg"], st["def_ppg"] = st["pf"] / n, st["pa"] / n
    st["pd_pg"] = st["off_ppg"] - st["def_ppg"]

    # ---- scrimmage plays (pass + run, excluding two-point tries) ----
    ps = pbp[pbp["game_id"].isin(ids) & pbp["play_type"].isin(["pass", "run"])
             & pbp["posteam"].notna() & pbp["defteam"].notna()
             & (pbp["two_point_attempt"].fillna(0) != 1)].copy()
    yds = ps["yards_gained"].fillna(0)
    is_pass = ps["play_type"] == "pass"
    ps["yds"], ps["pass_yds"], ps["rush_yds"] = yds, yds.where(is_pass, 0), yds.where(~is_pass, 0)
    ps["tov"] = (ps["interception"].fillna(0) == 1).astype(int) + (ps["fumble_lost"].fillna(0) == 1).astype(int)
    ps["sack_f"] = (ps["sack"].fillna(0) == 1).astype(int)
    ps["t3c"] = (ps["third_down_converted"].fillna(0) == 1).astype(int)
    ps["t3a"] = ps["t3c"] + (ps["third_down_failed"].fillna(0) == 1).astype(int)
    ps["expl"] = (((~is_pass) & (yds >= EXPL_RUN_YDS))
                  | (is_pass & (ps["sack"].fillna(0) != 1) & (yds >= EXPL_PASS_YDS))).astype(int)
    cols = ["yds", "pass_yds", "rush_yds", "tov", "sack_f", "t3c", "t3a", "expl"]
    off, dfn = ps.groupby("posteam")[cols].sum(), ps.groupby("defteam")[cols].sum()
    eps = ps[ps["epa"].notna()]
    pb = prepare_dropbacks(pbp[pbp["game_id"].isin(ids)])
    oa, da = pb.groupby("posteam")[["num", "den"]].sum(), pb.groupby("defteam")[["num", "den"]].sum()

    trips = red_zone_drives(pbp[pbp["game_id"].isin(ids)])
    ds = drive_starts(pbp[pbp["game_id"].isin(ids)])
    st["off_fp"] = 100 - ds.groupby("posteam")["yardline_100"].mean()  # own yard line where drives start
    st["def_fp"] = 100 - ds.groupby("defteam")["yardline_100"].mean()  # where opponents' drives start vs this defense
    st["off_rz"] = 100 * trips.groupby("posteam")["td"].mean()        # TD % of red-zone trips
    st["def_rz"] = 100 * trips.groupby("defteam")["td"].mean()        # TD % allowed

    for side, d, anya, epa_s in (("off", off, oa, eps.groupby("posteam")["epa"].mean()),
                                 ("def", dfn, da, eps.groupby("defteam")["epa"].mean())):
        st[f"{side}_ypg"], st[f"{side}_pass_ypg"], st[f"{side}_rush_ypg"] = d["yds"] / n, d["pass_yds"] / n, d["rush_yds"] / n
        st[f"{side}_3d"] = 100 * d["t3c"] / d["t3a"].where(d["t3a"] > 0)
        st[f"{side}_expl"] = d["expl"] / n
        st[f"{side}_epa"] = epa_s
        st[f"{side}_anya"] = anya["num"] / anya["den"].where(anya["den"] > 0)
    st["off_tov"], st["off_sacks"] = off["tov"], off["sack_f"]          # giveaways, sacks taken
    st["def_tov"], st["def_sacks"] = dfn["tov"], dfn["sack_f"]          # takeaways, sacks made
    st["tov_margin"] = st["def_tov"] - st["off_tov"]
    st["net_epa"] = st["off_epa"] - st["def_epa"]

    eps_s = eps.assign(s=(eps["epa"] > 0).astype(int))
    st["off_succ"] = 100 * eps_s.groupby("posteam")["s"].mean()
    st["def_succ"] = 100 * eps_s.groupby("defteam")["s"].mean()

    rush_eps = eps[eps["play_type"] == "run"]
    st["off_rush_epa"] = rush_eps.groupby("posteam")["epa"].mean()
    st["def_rush_epa"] = rush_eps.groupby("defteam")["epa"].mean()
    pr = pb.assign(pr=((pb["sk"] == 1) | (pb["qb_hit"].fillna(0) == 1)).astype(int))
    st["off_press"] = 100 * pr.groupby("posteam")["pr"].mean()        # pressure allowed by the offense
    st["def_press"] = 100 * pr.groupby("defteam")["pr"].mean()        # pressure generated by the defense
    sd = scoring_drives(pbp[pbp["game_id"].isin(ids)])
    st["off_drive"] = 100 * sd.groupby("posteam")["score"].mean() if sd is not None else np.nan
    st["def_drive"] = 100 * sd.groupby("defteam")["score"].mean() if sd is not None else np.nan

    for _, _, col, higher, _ in STAT_SPECS:
        st[col + "_rk"] = st[col].rank(ascending=not higher, method="min")
    return st


# ----------------------------------------------------------------------------
# 8b. KEY PLAYER MATCHUPS: WR1 vs the opposing top corner (display only, not a model input)
# ----------------------------------------------------------------------------
# Free NFL data does NOT say which cornerback covered which receiver, so true shadow coverage
# cannot be read from it. This panel shows each player's own numbers, whether they have been in
# the same game before, and an ESTIMATE of whether shadowing is plausible (elite corner vs a
# receiver who dominates his team's targets).
PM_MIN_CB_TARGETS = 20         # coverage numbers need at least this many targets to be ranked
PM_ELITE_CB_RANK = 8           # "elite" = top N corners in the league by passer rating allowed
PM_ALPHA_WR_SHARE = 0.25       # WR1 is an "alpha" when he sees this share of his team's targets


def load_player_info():
    """DataFrame gsis_id, pfr_id, name, position (empty frame if unavailable)."""
    try:
        pl = to_pandas(nfl.load_players())
        name_c = first_col(pl, ["display_name", "full_name", "name"])
        pos_c = first_col(pl, ["position", "pos"])
        return pd.DataFrame({
            "gsis_id": pl["gsis_id"],
            "pfr_id": pl["pfr_id"] if "pfr_id" in pl.columns else None,
            "name": pl[name_c] if name_c else pl["gsis_id"],
            "position": pl[pos_c].astype(str).str.upper() if pos_c else ""}).dropna(subset=["gsis_id"])
    except Exception as ex:
        print(f"  ! load_players failed ({ex}); player matchups skipped")
        return pd.DataFrame(columns=["gsis_id", "pfr_id", "name", "position"])


def load_depth_cbs():
    """{team: [(gsis_id, name), ...]} starting corners from the latest depth chart snapshot, plus its time."""
    try:
        dc = to_pandas(nfl.load_depth_charts(SEASONS))
        need = ["dt", "team", "gsis_id", "pos_abb", "pos_rank"]
        if any(c not in dc.columns for c in need):
            return {}, None
        dc = dc[dc["pos_abb"].astype(str).str.upper().isin(["LCB", "RCB", "CB"])].copy()
        dc["dt"] = pd.to_datetime(dc["dt"], utc=True, errors="coerce")
        dc["rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
        dc = dc[(dc["rank"] == 1)].dropna(subset=["dt", "gsis_id"])
        if dc.empty:
            return {}, None
        dc["team"] = fix_team(dc["team"])
        dc = dc[dc["dt"] == dc.groupby("team")["dt"].transform("max")]
        nm = "player_name" if "player_name" in dc.columns else "gsis_id"
        out = {t: list(dict.fromkeys(zip(d["gsis_id"], d[nm]))) for t, d in dc.groupby("team")}
        return out, dc["dt"].max()
    except Exception as ex:
        print(f"  ! depth chart corners unavailable ({ex}); CB matchups skipped")
        return {}, None


def load_pfr_def(info):
    """Per player-game pass-coverage stats (PFR advanced defense) keyed by gsis_id, or None."""
    try:
        d = to_pandas(nfl.load_pfr_advstats(SEASONS, stat_type="def", summary_level="week"))
    except Exception as ex:
        print(f"  ! PFR coverage stats unavailable ({ex}); corner coverage numbers will be blank")
        return None
    pid = first_col(d, ["pfr_player_id", "pfr_id"])
    cols = {k: first_col(d, c) for k, c in {
        "tgt": ["def_targets"], "cmp": ["def_completions_allowed"], "yds": ["def_yards_allowed"],
        "td": ["def_receiving_td_allowed"], "rtg": ["def_passer_rating_allowed"]}.items()}
    if pid is None or cols["tgt"] is None or not {"season", "week", "team"} <= set(d.columns):
        print("  ! PFR coverage stat columns not as expected; corner coverage numbers will be blank")
        return None
    out = pd.DataFrame({"pfr_id": d[pid], "season": d["season"], "week": d["week"], "team": fix_team(d["team"]),
                        "tgt": d[cols["tgt"]], "cmp": d[cols["cmp"]] if cols["cmp"] else np.nan,
                        "yds": d[cols["yds"]] if cols["yds"] else np.nan,
                        "td": d[cols["td"]] if cols["td"] else np.nan,
                        "rtg": d[cols["rtg"]] if cols["rtg"] else np.nan})
    m = dict(zip(info["pfr_id"], info["gsis_id"]))
    out["gsis_id"] = out["pfr_id"].map(m)
    return out.dropna(subset=["gsis_id"])


def build_player_matchups(pred, tg, pbp, sched):
    """{(home, away): html} for the upcoming week; {} if the needed data is missing."""
    info = load_player_info()
    if info.empty:
        return {}
    names = dict(zip(info["gsis_id"], info["name"]))
    pos = dict(zip(info["gsis_id"], info["position"]))
    cbs, snap = load_depth_cbs()
    pfr = load_pfr_def(info)
    wk = pred["week"]
    # who is Out / Doubtful this week
    status = {}
    try:
        inj = to_pandas(nfl.load_injuries(SEASONS))
        sc_ = first_col(inj, ["report_status", "game_status"])
        if sc_ is not None and "gsis_id" in inj.columns:
            inj = inj[(inj["season"] == PREDICT_SEASON) & (inj["week"] == wk)]
            status = {g: str(x).title() for g, x in zip(inj["gsis_id"], inj[sc_]) if str(x).title() in STATUS_FACTOR}
    except Exception:
        pass

    cur = pbp[pbp["season"] == PREDICT_SEASON]
    games_played = (sched[(sched["season"] == PREDICT_SEASON) & sched["result"].notna()])
    gp = {t: int(((games_played["home_team"] == t) | (games_played["away_team"] == t)).sum()) for t in pred_teams(pred)}
    tgt = cur[(cur["pass_attempt"] == 1) & (cur["sack"].fillna(0) != 1) & cur["receiver_player_id"].notna()]
    team_tgts = tgt.groupby("posteam").size().to_dict()
    wr1_of = tg[(tg["season"] == PREDICT_SEASON) & (tg["week"] == wk)].set_index("team")["wr1_id"].to_dict()

    # season coverage totals per player (this season) + league ranking of corners by passer rating allowed
    cov, cov_rank = {}, {}
    if pfr is not None:
        cs = pfr[pfr["season"] == PREDICT_SEASON].copy()
        cs["w_rtg"] = cs["rtg"] * cs["tgt"]
        agg = cs.groupby("gsis_id").agg(tgt=("tgt", "sum"), cmp=("cmp", "sum"), yds=("yds", "sum"),
                                        td=("td", "sum"), w=("w_rtg", "sum"), n=("tgt", "size"))
        agg["rtg"] = agg["w"] / agg["tgt"].where(agg["tgt"] > 0)
        cov = agg.to_dict("index")
        is_cb = pd.Series([pos.get(g, "CB") in ("CB", "DB") for g in agg.index], index=agg.index)
        q = agg[(agg["tgt"] >= PM_MIN_CB_TARGETS) & is_cb]
        cov_rank = q["rtg"].rank(method="min").to_dict()

    def wr_line(gid):
        d = tgt[tgt["receiver_player_id"] == gid]
        if d.empty:
            return None
        team = d["posteam"].mode().iat[0]
        ng = max(gp.get(team, 0), 1)
        return {"tg": len(d) / ng, "yd": d["yards_gained"].fillna(0).sum() / ng,
                "share": len(d) / max(team_tgts.get(team, 0), 1), "epa": d["epa"].mean()}

    def faced(wr_id, cb_id):
        """earlier games (any season loaded) where the WR had targets and the CB logged coverage stats."""
        if pfr is None:
            return None
        wg = tgt_all[tgt_all["receiver_player_id"] == wr_id]
        wgames = wg.groupby(["season", "week", "defteam"]).agg(t=("epa", "size"),
                                                                   y=("yards_gained", lambda x: x.fillna(0).sum())).reset_index()
        cg = pfr[pfr["gsis_id"] == cb_id][["season", "week", "team"]].rename(columns={"team": "defteam"})
        both = wgames.merge(cg, on=["season", "week", "defteam"])
        return both

    tgt_all = pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"].fillna(0) != 1) & pbp["receiver_player_id"].notna()]

    def tag(g):
        st = status.get(g)
        return f' <b style="color:#c0392b">({st.upper()})</b>' if st else ""

    def side_html(wr_team, cb_team, label):
        wr_id = wr1_of.get(wr_team)
        starters = cbs.get(cb_team, [])
        if not wr_id or not starters:
            return (f'<div class="panel"><h4>{html.escape(label)}</h4><p class="note">Not enough data '
                    f'(needs this team\'s WR1 and the other team\'s depth-chart corners).</p></div>')
        wr = wr_line(wr_id)
        # top corner = starting corner with the best coverage numbers (lowest passer rating allowed)
        def key(c):
            r = cov.get(c[0], {})
            return (r.get("rtg") if r.get("tgt", 0) >= 10 and pd.notna(r.get("rtg")) else 999)
        cb_id, cb_name = sorted(starters, key=key)[0]
        c = cov.get(cb_id)
        rk = cov_rank.get(cb_id)
        alpha = bool(wr and wr["share"] >= PM_ALPHA_WR_SHARE)
        elite = bool(rk is not None and rk <= PM_ELITE_CB_RANK)
        if alpha and elite:
            verdict, why = "Key matchup", "elite corner against a receiver who dominates his team's targets"
        elif alpha or elite:
            verdict = "Worth watching"
            why = "the receiver dominates his team's targets" if alpha else "the corner is among the league's best in coverage"
        else:
            verdict, why = "Low impact", "neither player stands out enough to build the game around"
        shadow = ("Shadow coverage is plausible (estimate): elite corner vs an alpha receiver."
                  if alpha and elite else "No strong sign of shadow coverage (estimate).")
        f = faced(wr_id, cb_id)
        if f is None:
            fl = "Earlier meetings: coverage data unavailable."
        elif f.empty:
            fl = "These two have not been in the same game before (2016 on)."
        else:
            season_ypt = (wr["yd"] / wr["tg"]) if wr and wr["tg"] else float("nan")
            fl = (f"Both played in {len(f)} earlier game(s) together (2016 on): the WR had {f['t'].sum() / len(f):.1f} "
                  f"targets and {f['y'].sum() / len(f):.0f} yards per game in those. This does not prove they were matched up.")
        wr_row = (f'<tr><td>WR1: <b>{html.escape(str(names.get(wr_id, wr_id)))}</b>{tag(wr_id)}</td>'
                  + (f'<td class="num">{wr["tg"]:.1f} tgt/g</td><td class="num">{wr["yd"]:.0f} yd/g</td>'
                     f'<td class="num">{wr["share"] * 100:.0f}% of targets</td><td class="num">{wr["epa"]:+.2f} EPA/tgt</td>'
                     if wr else '<td colspan="4" class="note">no 2026 targets yet</td>') + '</tr>')
        if c and c["tgt"] > 0:
            cb_cells = (f'<td class="num">{c["tgt"] / max(c["n"], 1):.1f} tgt/g</td>'
                        f'<td class="num">{100 * c["cmp"] / c["tgt"]:.0f}% comp</td>'
                        f'<td class="num">{c["yds"] / c["tgt"]:.1f} yd/tgt</td>'
                        f'<td class="num">{c["rtg"]:.0f} rating' + (f' (#{int(rk)})' if rk is not None else '') + '</td>')
        else:
            cb_cells = '<td colspan="4" class="note">coverage numbers unavailable</td>'
        cb_row = f'<tr><td>CB1: <b>{html.escape(str(cb_name))}</b>{tag(cb_id)}</td>{cb_cells}</tr>'
        return (f'<div class="panel"><h4>{html.escape(label)} &mdash; {verdict}</h4>'
                f'<div class="scroll"><table>{wr_row}{cb_row}</table></div>'
                f'<p class="note">{html.escape(verdict)}: {html.escape(why)}. {html.escape(shadow)}<br>{html.escape(fl)}</p></div>')

    out = {}
    for r in pred["rows"]:
        out[(r["home"], r["away"])] = (
            '<div class="cols" style="margin-top:16px">'
            + side_html(r["away"], r["home"], f'{r["away"]} WR1 vs {r["home"]} top corner')
            + side_html(r["home"], r["away"], f'{r["home"]} WR1 vs {r["away"]} top corner') + '</div>'
            '<p class="note">Estimates only: free NFL data does not record who covered whom, so shadow coverage '
            'is inferred from player quality and target share, not observed. Corner stats are pass-coverage '
            f'numbers for {PREDICT_SEASON}; corners come from the latest depth chart.</p>')
    print(f"  Player matchups built for {len(out)} games" + ("" if pfr is not None else " (no coverage stats)"))
    return out


def pred_teams(pred):
    return {t for r in pred["rows"] for t in (r["home"], r["away"])}



# ----------------------------------------------------------------------------
# 10. SHAREABLE HTML REPORT
# ----------------------------------------------------------------------------
FEATURE_LABELS = {
    "anya_off_diff": "Offensive ANY/A (home - away)",
    "anya_def_diff": "Defensive ANY/A allowed (home - away)",
    "epa_off_diff": "Offensive EPA/play (home - away)",
    "epa_def_diff": "Defensive EPA/play allowed (home - away)",
    "rk_home_off_vs_away_def": "Home offense rating vs away defense rating",
    "rk_away_off_vs_home_def": "Away offense rating vs home defense rating (negative expected)",
    "tov_margin_diff": "Turnover margin per game (home - away)",
    "expl_off_diff": "Offensive explosive-play rate (home - away)",
    "expl_def_diff": "Explosive-play rate allowed (home - away)",
    "rz_off_diff": "Offensive red-zone TD % (home - away)",
    "rz_def_diff": "Defensive red-zone TD % allowed (home - away)",
    "sos_diff": "Strength of schedule (home - away)",
    "sos_adj_net_epa_diff": "SOS-adjusted net EPA (home - away)",
    "common_opp_margin_diff": "Common-opponent margin (home - away)",
    "common_opp_epa_diff": "Common-opponent net EPA (home - away)",
    "has_common_opponents": "Has common opponents (flag)",
    "elo_diff": "Elo rating (home - away)",
    "elo2_diff": "EPA-based Elo rating (home - away)",
    "qb_q_diff": "Starting QB quality, EPA/dropback (home - away)",
    "qb_pr_diff": "Starting QB passer rating (home - away)",
    "wx_wind_x_qb": "Wind x QB-quality gap",
    "aggr_diff": "4th-down aggressiveness (home - away)",
    "cont_early": "Returning offensive production (home - away, early season)",
    "newqb_early": "New starting QB (home - away, early season)",
    "div_x_elo": "Divisional game x Elo gap",
    "m_pass": "Pass offense vs pass defense matchup (home - away)",
    "m_rush": "Rush offense vs rush defense matchup (home - away)",
    "m_press": "Pass rush vs pass protection matchup (home - away)",
    "div_game": "Divisional game (flag)",
    "qb_out_diff": "Starting QB out (away - home)",
    "wr_lost_diff": "Share of targets belonging to injured receivers (away - home)",
    "injury_score_diff": "Weighted injury score (away - home)",
    "home_field": "Home-field indicator",
    "hfa_diff": "Team home-field advantage (home - away)",
}

REPORT_CSS = """
:root{color-scheme:light dark;--goodbg:rgba(10,125,60,.13);--badbg:rgba(180,35,24,.11);--bg:#f6f7f9;--card:#fff;--text:#161a1f;--muted:#667085;
--line:#e4e7ec;--accent:#1d4ed8;--pos:#0a7d3c;--neg:#b42318;--bar:#dbe4ff;--barfav:#1d4ed8;
--winbg:rgba(29,78,216,.10)}
@media (prefers-color-scheme:dark){:root{--goodbg:rgba(74,222,128,.15);--badbg:rgba(248,113,113,.14);--bg:#0f1318;--card:#171c23;--text:#e8ebef;
--muted:#98a2b3;--line:#2a313b;--accent:#7aa2ff;--pos:#4ade80;--neg:#f87171;--bar:#27324a;
--barfav:#7aa2ff;--winbg:rgba(122,162,255,.16)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1050px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 10px}
h4{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.sub{color:var(--muted);margin:0 0 6px}
.toolbar{display:flex;gap:8px;margin:14px 0 12px}
.toolbar button{background:var(--card);color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}
.games{display:flex;flex-direction:column;gap:12px}
details.game,details.section{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
details.game>summary{list-style:none;cursor:pointer;padding:14px 16px;display:grid;
grid-template-columns:minmax(170px,1.4fr) minmax(100px,.8fr) minmax(150px,1fr) minmax(170px,1.2fr) minmax(190px,1fr) 20px;
gap:10px 18px;align-items:center}
details>summary::-webkit-details-marker{display:none}
.teams{font-weight:700;font-size:17px}.at{color:var(--muted);margin:0 6px;font-weight:400}
.rec{display:block;color:var(--muted);font-size:12px;font-weight:400}
.k{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.who{font-size:19px;font-weight:800;color:var(--accent)}
.score{white-space:nowrap;font-weight:600}.score .n{font-size:19px;font-weight:700;color:var(--muted)}
.score .n.w{color:var(--accent);font-weight:800}
.pct{font-size:20px;font-weight:800}.tier{color:var(--muted);font-size:12px;margin-left:8px}
.bar{display:block;height:8px;background:var(--bar);border-radius:4px;overflow:hidden;margin-top:5px}
.fill{display:block;height:100%;background:var(--barfav)}
.chev{color:var(--muted);transition:transform .15s}
.fin{display:block;white-space:normal;font-size:12px;font-weight:700;margin-top:3px}.fin.ok{color:#2e9e5b}.fin.no{color:#c0392b}.fin.tie{color:var(--muted)}
.vegas{text-align:right;min-width:0}.vegas .vr{display:block;white-space:nowrap;font-size:14px;color:var(--muted)}
.vegas .vr.pk{color:#2e9e5b;font-weight:800;font-size:15px}
.vegas .sub{display:block;color:var(--muted);font-size:12px;margin-top:3px}
.vegas .flag{display:inline-block;white-space:nowrap;margin-top:4px;padding:1px 7px;border-radius:999px;background:#d98a1f22;color:#d98a1f;font-size:11.5px;font-weight:700}
.imp.high{color:#c0392b;font-weight:800}.imp.medium{color:#d98a1f;font-weight:700}.imp.low,.imp.none,.imp.not{color:var(--muted)}
.wxbox{border:1px solid var(--line);border-radius:10px;padding:9px 12px;margin:0 0 10px;font-size:13.5px;line-height:1.45}
.wxbox .wxsrc{color:var(--muted);font-size:12px}
.whybox{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 12px}
.whybox ul{margin:6px 0 0;padding-left:18px}.whybox li{margin:5px 0;font-size:13.5px;line-height:1.4}
.shifts{display:flex;flex-wrap:wrap;gap:5px}.shift{font-size:12px;padding:2px 8px;border-radius:999px;background:var(--bar);color:var(--muted)}
.shift.up{background:rgba(46,158,91,.16);color:#2e9e5b}.shift.dn{background:rgba(192,57,43,.14);color:#c0392b}
.whyconf{margin:0 0 4px;font-size:14px}.whyhist{margin:8px 0 0;font-size:12px;color:var(--muted)}
@media (max-width:760px){.whybox{grid-template-columns:1fr}}
.stylebox{margin:0 0 10px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;font-size:13px}
.stylebox b{color:var(--fg)}.stylebox div{margin:2px 0;color:var(--muted)}
.edges{margin:0 0 14px;padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.edges h3{margin:0 0 6px;font-size:14px}.edges div{font-size:13px;color:var(--muted);margin:2px 0}.edges b{color:var(--fg)}
.vegas .ag{display:block;font-size:12px;font-weight:700;margin-top:2px}.ag.yes{color:#2e9e5b}.ag.no{color:#d98a1f}
.tl{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.logo{display:inline-block;flex:none;width:22px;height:22px;background-size:contain;background-repeat:no-repeat;background-position:center}
.logo.lg{width:30px;height:30px}
th.tm .tl{justify-content:center}
details[open]>summary .chev{transform:rotate(180deg)}
.detail{border-top:1px solid var(--line);padding:14px 16px 16px}
.probline{color:var(--muted);font-size:13px;margin-bottom:12px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.panel{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
table{border-collapse:collapse;width:100%}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
table.cmp td.v{text-align:center;width:34%;font-variant-numeric:tabular-nums}
table.cmp td.v.good{font-weight:700;background:var(--goodbg);color:var(--pos)}
table.cmp td.v.bad{background:var(--badbg);color:var(--neg)}
table.cmp td.lbl{text-align:center;color:var(--muted);font-size:12px}
table.cmp th.tm{text-align:center;font-size:13px;color:var(--text)}
table.cmp tr.sec td{background:var(--bg);color:var(--muted);font-size:11px;font-weight:700;
text-transform:uppercase;letter-spacing:.06em;text-align:left}
.rk{font-size:11px;margin-left:5px;font-weight:600}
.rk.good{color:var(--pos)}.rk.bad{color:var(--neg)}.rk.mid{color:var(--muted)}
.scroll{overflow-x:auto}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
details.section{margin-top:14px}
details.section>summary{cursor:pointer;list-style:none;padding:12px 16px;font-weight:700}
details.section>summary::after{content:" \\25BE";color:var(--muted)}
details.section .inner{padding:0 16px 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat .v{font-size:22px;font-weight:700}.stat .l{color:var(--muted);font-size:12px}
.cbar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}
.cbar.p{background:var(--pos)}.cbar.n{background:var(--neg)}
.note{color:var(--muted);font-size:13px;margin-top:10px}
.warn{background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.5);border-radius:8px;
padding:8px 12px;margin:10px 0;font-size:13px}
@media (max-width:640px){
details.game>summary{grid-template-columns:1fr 1fr}.teams{grid-column:1/-1}.chev{display:none}
.cols{grid-template-columns:1fr}}
"""


def am_to_dec(am):
    """American odds -> decimal payout multiple (stake included)."""
    am = float(am)
    return 1 + am / 100.0 if am > 0 else 1 + 100.0 / abs(am)


def dec_to_am(dec):
    """Decimal payout multiple -> American odds (rounded)."""
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100.0 / (dec - 1)))


def _market_legs(pred):
    """Every unplayed game x side that has a moneyline: model probability vs the market's no-vig probability."""
    legs = []
    for gi, r in enumerate(pred["rows"]):
        if r.get("played"):
            continue
        mh, ma, vp = r.get("v_ml_home"), r.get("v_ml_away"), r.get("v_prob")
        if any(x is None or pd.isna(x) for x in (mh, ma, vp)):
            continue
        for side in ("home", "away"):
            p = r["p_home"] if side == "home" else 1 - r["p_home"]
            mk = vp if side == "home" else 1 - vp
            ml = mh if side == "home" else ma
            legs.append(dict(g=gi, team=r[side], opp=r["away" if side == "home" else "home"], home=(side == "home"),
                             p=float(p), mk=float(mk), ml=float(ml), dec=am_to_dec(ml), edge=float(p - mk), pick=bool(p >= 0.5)))
    return legs


PARLAY_TIERS = [
    # label, blurb, legs, leg filter, sort key
    ("Safest 2-leg", "The two most likely winners together. Small payout, best chance to cash.", 2,
     lambda l: l["p"] >= 0.60, "p"),
    ("2-leg value", "Two picks where the model rates the team higher than the betting price implies.", 2,
     lambda l: l["edge"] >= 0.01 and l["p"] >= 0.55, "ev"),
    ("3-leg value", "Three picks the model likes at their price, balancing payout and chance.", 3,
     lambda l: l["edge"] >= 0.0 and l["p"] >= 0.50, "ev"),
    ("4-leg stretch", "Bigger payout; all four have to hit, and underdogs can sneak in.", 4,
     lambda l: l["edge"] >= 0.0 and l["p"] >= 0.45, "ev"),
    ("5-leg longshot", "Lottery-ticket odds. Expect to lose most of the time.", 5,
     lambda l: l["edge"] >= 0.0 and l["p"] >= 0.40, "ev"),
]


def build_parlays(pred, per_tier=2):
    """Parlays from the model's win probabilities, grouped by risk. Games are treated as independent."""
    from itertools import combinations
    from math import prod
    legs = _market_legs(pred)
    cands = [l for l in legs if l["pick"] or (l["edge"] >= 0.03 and l["p"] >= 0.40)]
    cands.sort(key=lambda l: -(l["p"] + l["edge"]))
    cands = cands[:16]
    out = []
    for label, blurb, n, ok, key in PARLAY_TIERS:
        pool = [l for l in cands if ok(l)]
        res = []
        for c in combinations(pool, n):
            if len({l["g"] for l in c}) < n:
                continue
            p, dec = prod(l["p"] for l in c), prod(l["dec"] for l in c)
            res.append(dict(legs=c, p=p, dec=dec, ev=p * dec - 1))
        res.sort(key=lambda x: -x[key])
        chosen = []
        for x in res:
            ids = {(l["g"], l["team"]) for l in x["legs"]}
            if all(len(ids & {(l["g"], l["team"]) for l in y["legs"]}) <= max(1, n - 2) for y in chosen):
                chosen.append(x)
            if len(chosen) >= per_tier:
                break
        if chosen:
            out.append(dict(label=label, blurb=blurb, n=n, parlays=chosen))
    return out


def find_underdogs(pred, min_prob=0.40, min_edge=0.04):
    """Market underdogs the model gives a real chance: model >= min_prob and ahead of the market by min_edge (or model pick)."""
    d = [l for l in _market_legs(pred) if l["mk"] < 0.5 and l["p"] >= min_prob and (l["edge"] >= min_edge or l["pick"])]
    d.sort(key=lambda l: -l["edge"])
    return d


REPORT_CSS2 = """
.wrap{max-width:1440px}
.toptabs{display:flex;gap:6px;margin:14px 0 14px;border-bottom:1px solid var(--line);overflow-x:auto;scrollbar-width:none}.toptabs::-webkit-scrollbar{display:none}
.toptab{flex:0 0 auto;white-space:nowrap;background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);font:inherit;font-weight:700;padding:8px 14px;cursor:pointer}
.toptab.on{color:var(--accent);border-bottom-color:var(--accent)}
.toppane[hidden],.gdetail[hidden],.tp[hidden]{display:none}
.app{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;align-items:start}
.glist{position:sticky;top:10px;max-height:calc(100vh - 20px);overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding-right:2px}
.gitem{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center;text-align:left;background:var(--card);color:var(--text);
border:1px solid var(--line);border-left:4px solid transparent;border-radius:10px;padding:9px 11px;font:inherit;cursor:pointer}
.gitem:hover{border-color:var(--accent)}
.gitem.on{border-color:var(--accent);border-left-color:var(--accent);background:var(--winbg)}
.gi-teams{font-weight:700;font-size:14px;white-space:nowrap}.gi-teams .at{margin:0 5px}
.gi-pick{grid-column:1;font-size:12.5px;color:var(--muted)}.gi-pick b{color:var(--text)}
.chip{grid-column:2;grid-row:1/3;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--bar);color:var(--muted);white-space:nowrap}
.chip.ok{background:rgba(46,158,91,.18);color:#2e9e5b}.chip.no{background:rgba(192,57,43,.16);color:#c0392b}.chip.flag{background:#d98a1f22;color:#d98a1f}
.gview{background:var(--card);border:1px solid var(--line);border-radius:14px;min-width:0}
.ghead{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px 22px;align-items:start;padding:16px;border-bottom:1px solid var(--line)}
.ghead .teams{grid-column:1/-1}
.subtabs{display:flex;gap:4px;padding:8px 12px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
.subtabs button{background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);font:inherit;font-weight:600;padding:7px 12px;cursor:pointer}
.subtabs button.on{color:var(--accent);border-bottom-color:var(--accent)}
.tp{padding:14px 16px 16px}
.subtabs .cnt{display:inline-block;margin-left:4px;padding:0 6px;border-radius:999px;background:rgba(192,57,43,.16);color:#c0392b;font-size:11px;font-weight:700}
.st.out{color:#c0392b;font-weight:700}.st.doubtful{color:#d98a1f;font-weight:700}.st.questionable{color:#b8a21f;font-weight:600}
.pr.dnp{color:#c0392b;font-weight:700}.pr.limited{color:#d98a1f;font-weight:600}.pr.full{color:#2e9e5b}
table.mp td{vertical-align:middle}.mp-row{cursor:pointer}.mp-row:hover td{background:var(--winbg)}
.mp-pick .tl{font-weight:800;color:var(--accent)}
.mp-conf{min-width:150px}.mp-conf .bar{margin-top:3px;max-width:140px}
.mp-sc{white-space:nowrap}.mp-v{font-size:13px;color:var(--muted);white-space:nowrap}.mp-v.far{color:#d98a1f;font-weight:700}
.mp-res{font-weight:700;white-space:nowrap}.mp-res.ok{color:#2e9e5b}.mp-res.no{color:#c0392b}.mp-res.tie{color:var(--muted)}
.dogs{margin-top:22px}.dogs h3,.pltier h3{margin:0 0 4px;font-size:17px}
.dogs .sub2,.pltier .sub2{margin:0 0 10px;font-size:13px;color:var(--muted)}
.pl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;margin-bottom:22px}
.plcard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.pl-leg{display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:7px 0;border-bottom:1px solid var(--line);align-items:center}
.pl-leg .pl-vs{color:var(--muted);font-size:12.5px;margin-left:6px}
.pl-ml{font-weight:800;text-align:right}.pl-m{grid-column:1/-1;font-size:12px;color:var(--muted)}
.pl-foot{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-top:10px;font-size:13px}
.pl-foot b{font-size:15px}.pl-foot .k{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.pl-pos{color:#2e9e5b}.pl-neg{color:#c0392b}
.pl-note{font-size:12.5px;color:var(--muted);margin:0 0 14px}
.secgroup>details.section{border:0;background:none;margin:0}
.secgroup>details.section>summary{display:none}
.secgroup>details.section>.inner{padding:0}
.secnav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.secnav button{background:var(--card);color:var(--muted);border:1px solid var(--line);border-radius:999px;font:inherit;font-size:13px;font-weight:600;padding:5px 13px;cursor:pointer}
.secnav button.on{background:var(--winbg);color:var(--accent);border-color:var(--accent)}
@media (max-width:860px){.app{grid-template-columns:minmax(0,1fr)}.glist{position:static;flex-direction:row;overflow-x:auto;max-height:none;padding-bottom:6px}
.gitem{flex:0 0 auto;width:max-content;min-width:176px;grid-template-columns:auto;padding-right:14px}
.gitem .chip{grid-column:1;grid-row:auto;justify-self:start;margin-top:3px}
.glist{scroll-snap-type:x proximity}.gitem{scroll-snap-align:start}}
"""

REPORT_JS = """
(function(){
  function $(s,r){return (r||document).querySelector(s)}
  function $$(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))}
  var tabs=$$('.toptab'), panes=$$('.toppane');
  function showTop(id){tabs.forEach(function(b){b.classList.toggle('on',b.dataset.p===id)});
    panes.forEach(function(p){p.hidden=(p.id!==id)})}
  tabs.forEach(function(b){b.addEventListener('click',function(){showTop(b.dataset.p)})});
  showTop('p-games');

  var items=$$('.gitem'), secs=$$('.gdetail'), curTab='over', cur=-1;
  function applyTab(s){
    var ps=$$('.tp',s), has=ps.some(function(p){return p.dataset.t===curTab});
    var t=has?curTab:'over';
    $$('.subtabs button',s).forEach(function(b){b.classList.toggle('on',b.dataset.t===t)});
    ps.forEach(function(p){p.hidden=(p.dataset.t!==t)});
  }
  function showGame(i){
    if(i<0||i>=items.length)return; cur=i;
    items.forEach(function(b,k){b.classList.toggle('on',k===i)});
    secs.forEach(function(s){s.hidden=(s.id!=='g'+i)});
    applyTab($('#g'+i));
    if(window.innerWidth<=860){var g=$('.gview'); if(g&&g.scrollIntoView)g.scrollIntoView({block:'start'})}
  }
  items.forEach(function(b,k){b.addEventListener('click',function(){showGame(k)})});
  secs.forEach(function(s){$$('.subtabs button',s).forEach(function(b){
    b.addEventListener('click',function(){curTab=b.dataset.t; applyTab(s)})})});
  var first=0; for(var k=0;k<items.length;k++){if(items[k].dataset.played==='0'){first=k;break}}
  showGame(first);
  document.addEventListener('keydown',function(ev){
    if(ev.target&&/input|textarea|select/i.test(ev.target.tagName))return;
    if($('#p-games').hidden)return;
    if(ev.key==='ArrowDown'||ev.key==='ArrowUp'){ev.preventDefault(); showGame(cur+(ev.key==='ArrowDown'?1:-1));
      if(items[cur]&&items[cur].scrollIntoView)items[cur].scrollIntoView({block:'nearest'})}});

  // model-picks summary: tap a row to open that game
  $$('.mp-row').forEach(function(tr){tr.addEventListener('click',function(){
    showTop('p-games'); showGame(parseInt(tr.dataset.g,10));
    var g=$('.gview'); if(g&&g.scrollIntoView&&window.innerWidth<=860)g.scrollIntoView({block:'start'})})});

  // analysis groups: turn each drop-down section into a pill tab
  $$('.secgroup').forEach(function(grp){
    var ds=$$('details.section',grp).filter(function(d){return d.parentNode===grp});
    if(!ds.length)return;
    var nav=document.createElement('div'); nav.className='secnav';
    function pick(n){ds.forEach(function(d,k){d.hidden=(k!==n); d.open=true});
      $$('button',nav).forEach(function(b,k){b.classList.toggle('on',k===n)})}
    ds.forEach(function(d,k){var sm=$('summary',d), b=document.createElement('button');
      b.type='button'; b.textContent=(sm?sm.textContent:'Section '+(k+1)).replace(/\\s*\\u25BE\\s*$/,'');
      b.addEventListener('click',function(){pick(k)}); nav.appendChild(b)});
    grp.insertBefore(nav,grp.firstChild); pick(0);
  });
})();
"""


def build_injury_reports(pred):
    """
    This week's injury report for every team in the games shown: player, position, injury, practice status and
    game status, worst first. Returns {team: [row dicts]}. The data has one row per player per week, so the
    practice column is the LAST practice report of the week (usually Friday). Empty dict if unavailable.
    """
    try:
        inj = to_pandas(nfl.load_injuries(SEASONS))
    except Exception as ex:
        print(f"  ! injury report unavailable ({ex})")
        return {}
    if INJ_CARRIED_FROM:                                   # test mode: each team's latest report stands in for this week's
        inj = inj[inj["season"] == PREDICT_SEASON].copy()
        inj["team"] = fix_team(inj["team"])
        inj = latest_team_reports(inj, pred["week"]).copy()
    else:
        inj = inj[(inj["season"] == PREDICT_SEASON) & (inj["week"] == pred["week"])].copy()
    inj["team"] = fix_team(inj["team"])
    inj = add_manual_injuries(inj, pred["week"])
    if inj.empty:
        return {}
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    gs_c = first_col(inj, ["report_status", "game_status"])
    pr_c = first_col(inj, ["practice_status"])
    nm_c = first_col(inj, ["full_name", "player_name", "name"])
    if nm_c is None and {"first_name", "last_name"} <= set(inj.columns):
        inj["_nm"] = inj["first_name"].fillna("") + " " + inj["last_name"].fillna("")
        nm_c = "_nm"
    ij_c = first_col(inj, ["report_primary_injury", "practice_primary_injury", "injury"])
    if nm_c is None or (gs_c is None and pr_c is None):
        print("  ! injury report columns not recognized; skipped")
        return {}
    inj["team"] = fix_team(inj["team"])

    def short_practice(v):
        t = str(v).lower()
        if t in ("nan", "none", ""):
            return ""
        return "DNP" if "did not" in t else ("Limited" if "limited" in t else ("Full" if "full" in t else str(v)))
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    out = {}
    for t, g in inj.groupby("team"):
        rows = []
        for _, x in g.iterrows():
            gs = "" if gs_c is None or pd.isna(x[gs_c]) else str(x[gs_c]).strip().title()
            pr = short_practice(x[pr_c]) if pr_c is not None else ""
            if not gs and pr in ("", "Full"):
                continue                                      # a full-practice player with no game designation is not news
            w_ = INJ_IMPACT_NOW.get((t, x.get("gsis_id")))
            if w_ is None:
                imp = "not counted" if gs not in ("Out", "Doubtful") else "-"
            else:
                imp = "High" if w_ >= 1.5 else ("Medium" if w_ >= 0.6 else ("Low" if w_ > 0 else "None"))
            rows.append({"name": str(x[nm_c]).strip(), "pos": str(x.get("position", "")).upper(),
                         "injury": "" if ij_c is None or pd.isna(x[ij_c]) else str(x[ij_c]),
                         "practice": pr, "status": gs, "impact": imp})
        rows.sort(key=lambda r: (order.get(r["status"], 3), 0 if r["pos"] == "QB" else 1, r["pos"], r["name"]))
        out[t] = rows
    return out


def write_html_report(pred, report, elo_ratings, season_stats=None, logos=None, unit_table=None, player_html=None, inj_reports=None):
    """
    One self-contained HTML file (no internet needed): each game is a card showing
    the pick and confidence, with a drop-down that compares every stat side by side.
    """
    e = html.escape
    st = season_stats
    n_teams = len(st) if st is not None else 32
    logos = logos or {}
    logo_css = "".join(f'.logo-{e(t)}{{background-image:url("{u}")}}' for t, u in logos.items())

    def tm(team, big=False):
        """Team abbreviation with its logo (when one is available)."""
        t = e(team)
        if team in logos:
            return f'<span class="tl"><span class="logo{" lg" if big else ""} logo-{t}"></span>{t}</span>'
        return t

    def fmt(v, f):
        return "-" if v is None or pd.isna(v) else f.format(v)

    def better(a, h, higher=True):
        """'a' / 'h' for the better side, or None when equal / unknown."""
        if a is None or h is None or pd.isna(a) or pd.isna(h) or abs(a - h) < 1e-9:
            return None
        return "a" if (a > h) == higher else "h"

    def cmp_row(label, a_html, h_html, win=None):
        ca = "" if win is None else ("good" if win == "a" else "bad")     # green = stronger side
        ch = "" if win is None else ("good" if win == "h" else "bad")     # red = weaker side
        return (f'<tr><td class="v {ca}">{a_html}</td>'
                f'<td class="lbl">{e(label)}</td>'
                f'<td class="v {ch}">{h_html}</td></tr>')

    def rank_badge(rk):
        if rk is None or pd.isna(rk):
            return ""
        cls = "good" if rk <= n_teams / 3 else ("bad" if rk > 2 * n_teams / 3 else "mid")
        return f'<span class="rk {cls}">#{int(rk)}</span>'

    def model_table(r):
        spec = [("Elo rating", "elo", True, "{:.0f}"),
                ("ANY/A, offense (weighted avg)", "anya_off", True, "{:.2f}"),
                ("ANY/A allowed, defense", "anya_def", False, "{:.2f}"),
                ("EPA/play, offense (last 6)", "epa_off", True, "{:+.3f}"),
                ("EPA/play allowed, defense (last 6)", "epa_def", False, "{:+.3f}"),
                ("Success rate, offense", "succ_off", True, "{:.1f}%"),
                ("Success rate allowed, defense", "succ_def", False, "{:.1f}%"),
                ("Explosive-play rate, offense", "expl_off", True, "{:.1f}%"),
                ("Explosive-play rate allowed, defense", "expl_def", False, "{:.1f}%"),
                ("Turnover margin / game (last 8)", "tov", True, "{:+.2f}"),
                ("Red-zone TD %, offense (last 8 games)", "rz_off", True, "{:.1f}%"),
                ("Red-zone TD % allowed, defense", "rz_def", False, "{:.1f}%"),
                ("Net EPA/play", "net", True, "{:+.3f}"),
                ("Strength of schedule so far", "sos", None, "{:+.3f}"),
                ("SOS-adjusted net EPA", "sosadj", True, "{:+.3f}"),
                ("Home-field edge rating (pts)", "hfa", True, "{:.1f}"),
                ("Injury impact (higher = worse)", "inj", False, "{:.1f}")]
        rows = [cmp_row(lbl, fmt(r["a_" + k], f), fmt(r["h_" + k], f),
                        None if hi is None else better(r["a_" + k], r["h_" + k], hi))
                for lbl, k, hi, f in spec]
        rows.append(cmp_row("Starting QB", "OUT" if r["a_qb"] else "OK", "OUT" if r["h_qb"] else "OK",
                            better(r["a_qb"], r["h_qb"], False)))
        if r["n_common"] > 0:
            rows.append(cmp_row(f"Avg margin vs {r['n_common']} common opp.", fmt(r["a_cm"], "{:+.1f}"),
                                fmt(r["h_cm"], "{:+.1f}"), better(r["a_cm"], r["h_cm"], True)))
            rows.append(cmp_row("Net EPA/play vs common opp.", fmt(r["a_ce"], "{:+.3f}"),
                                fmt(r["h_ce"], "{:+.3f}"), better(r["a_ce"], r["h_ce"], True)))
        else:
            rows.append(cmp_row("Common opponents", "none yet", "none yet"))
        return (f'<table class="cmp"><tr><th class="tm">{tm(r["away"])}</th><th></th>'
                f'<th class="tm">{tm(r["home"])}</th></tr>{"".join(rows)}</table>')

    def records_table(r):
        """Records plus overall team quality, side by side."""
        if st is None or r["away"] not in st.index or r["home"] not in st.index:
            return '<p class="note">Season stats are not available yet.</p>'
        a, h = st.loc[r["away"]], st.loc[r["home"]]
        rows = [cmp_row("Overall record", e(a["record"]), e(h["record"])),
                cmp_row("At home", e(a["home_record"]), e(h["home_record"])),
                cmp_row("On the road", e(a["away_record"]), e(h["away_record"]))]
        for label, col, f in (("Point differential / game", "pd_pg", "{:+.1f}"),
                              ("Turnover margin", "tov_margin", "{:+.0f}"),
                              ("Net EPA / play", "net_epa", "{:+.3f}")):
            ra, rh = a[col + "_rk"], h[col + "_rk"]
            rows.append(cmp_row(label, fmt(a[col], f) + rank_badge(ra), fmt(h[col], f) + rank_badge(rh),
                                better(ra, rh, higher=False)))
        return (f'<table class="cmp"><tr><th class="tm">{tm(r["away"])}</th><th></th>'
                f'<th class="tm">{tm(r["home"])}</th></tr>{"".join(rows)}</table>')

    def injury_panel(team):
        rows = (inj_reports or {}).get(team, [])
        if not rows:
            return (f'<div class="panel"><h4>{tm(team)} injury report</h4>'
                    '<p class="note">No players listed yet (reports usually appear midweek).</p></div>')
        body = "".join(
            f'<tr><td><b>{e(x["name"])}</b></td><td>{e(x["pos"])}</td><td>{e(x["injury"])}</td>'
            f'<td><span class="pr {e(x["practice"].lower())}">{e(x["practice"] or "-")}</span></td>'
            f'<td><span class="st {e(x["status"].lower())}">{e(x["status"] or "-")}</span></td>'
            f'<td class="imp {e(x.get("impact", "").split()[0].lower() if x.get("impact") else "")}">{e(x.get("impact", ""))}</td></tr>' for x in rows)
        carried = (f'<p class="note" style="margin:0 0 8px">Test mode: this is the week {INJ_CARRIED_FROM} report, reused for this week.</p>'
                   if INJ_CARRIED_FROM else "")
        return (f'<div class="panel"><h4>{tm(team)} injury report</h4>{carried}<div class="scroll"><table><tr><th>Player</th><th>Pos</th>'
                f'<th>Injury</th><th>Practice</th><th>Game status</th><th>Model impact</th></tr>{body}</table></div></div>')

    def style_html(r):
        n = r.get("style") or []
        if not n:
            return ""
        return ('<div class="stylebox"><b>Notable matchups</b> <span style="font-size:11.5px">(last 8 games, vs league average; '
                'shown for context, not used in the pick)</span>' + "".join(f"<div>&bull; {e(t)}</div>" for t, _ in n) + "</div>")

    def weather_box(r):
        w = r.get("wx")
        if not w:
            return ('<div class="wxbox"><b>Weather</b> no forecast available; the model treats this game as calm and mild.</div>'
                    if not r.get("played") and r.get("roof_") not in ("dome", "closed") else "")
        if w.get("dome"):
            return '<div class="wxbox"><b>Weather</b> indoors (dome or closed roof), so weather does not matter.</div>'
        t_, wd_ = w["temp"], w["wind"]
        tags = []
        if wd_ >= 15:
            tags.append("windy: hurts passing and kicking")
        elif wd_ >= 10:
            tags.append("breezy")
        if t_ <= 32:
            tags.append("freezing")
        elif t_ <= 40:
            tags.append("cold")
        elif t_ >= 85:
            tags.append("hot")
        if w.get("rain") is not None and w["rain"] >= 50:
            tags.append(f"{w['rain']:.0f}% chance of precipitation")
        gust = f", gusts to {w['gust']:.0f}" if w.get("gust") and w["gust"] >= wd_ + 5 else ""
        src = "recorded at the game" if w.get("recorded") else "forecast at kickoff"
        tag = f' &middot; <b>{e("; ".join(tags))}</b>' if tags else " &middot; no weather concerns"
        return (f'<div class="wxbox"><b>Weather</b> {t_:.0f}&deg;F, wind {wd_:.0f} mph{e(gust)}{tag} '
                f'<span class="wxsrc">({src})</span></div>')

    def why_html(r):
        w = r.get("why")
        if not w:
            return ""
        li = lambda xs: "".join(f"<li>{e(t)}</li>" for t in xs)
        hist = f'<p class="whyhist">{e(w["hist"])}</p>' if w["hist"] else ""
        chips = "".join(f'<span class="shift {"up" if v_ >= 0.05 else ("dn" if v_ <= -0.05 else "")}">{e(n_)} {v_:+.1f}</span>' for n_, v_ in w.get("shifts", []))
        if chips:
            hist += ('<p class="whyhist" style="margin-bottom:4px">How far each factor moves the pick, in points of win probability '
                     f'versus an average game (+ helps {e(w["pick"])}):</p><div class="shifts">{chips}</div>')
        return (f'<div class="whybox"><div class="panel"><h4>Why the model picks {e(w["pick"])}</h4>'
                f'<p class="whyconf"><b>{e(w["word"])}</b> &middot; {w["pp"] * 100:.0f}% chance</p><ul>{li(w["reasons"])}</ul>{hist}</div>'
                f'<div class="panel"><h4>Concerns about this pick</h4><ul>{li(w["concerns"])}</ul>'
                '<p class="whyhist">The model sees team ratings, matchups, quarterbacks and who is out, weighted by position, how much each player plays, '
                'and (for pass rushers, linebackers, receivers and corners) how productive he has been. It cannot rate offensive linemen or safeties '
                'beyond their snaps, and does not know scheme or coaching changes, or motivation.</p></div></div>')

    def vegas_box(r):
        """Right-hand summary box: spread and moneyline for both teams (the model's pick in green),
        the total, whether the model agrees on the winner, and how the spread turned out once final."""
        sp = r.get("v_spread")
        if sp is None or pd.isna(sp):
            return '<div class="vegas"><span class="k">Vegas</span><span class="sub">no line yet</span></div>'

        def ml(v):
            return "" if v is None or pd.isna(v) else f" &middot; ML {int(v):+d}"

        def num(x):
            return f"{x:+g}" if x != 0 else "PK"
        pick_home = r["winner"] == r["home"]
        a_row = (f'<span class="vr{"" if pick_home else " pk"}">{e(r["away"])} {num(sp)}{ml(r.get("v_ml_away"))}</span>')
        h_row = (f'<span class="vr{" pk" if pick_home else ""}">{e(r["home"])} {num(-sp)}{ml(r.get("v_ml_home"))}</span>')
        tot = r.get("v_total")
        sub = f'<span class="sub">Total {tot:g}</span>' if tot is not None and not pd.isna(tot) else ""
        ag = ""
        fav = None if sp == 0 else (r["home"] if sp > 0 else r["away"])
        m_model = r["score_h"] - r["score_a"]
        if fav is not None:
            agree = fav == r["winner"]
            ag = (f'<span class="ag {"yes" if agree else "no"}">'
                  f'{"Model agrees on the winner" if agree else "Model picks the underdog"}</span>')
        # model spread (rounded to the nearest half point)
        pm = r["pred_margin"]
        ps = round(pm * 2) / 2
        mod = "PK" if ps == 0 else f'{e(r["home"] if ps > 0 else r["away"])} {-abs(ps):g}'
        mm = f'<span class="sub">Model spread: <b>{mod}</b></span>'
        ats = ""
        if r.get("played"):
            marg = int(r["final_h"]) - int(r["final_a"])
            if marg == sp:
                ats = '<span class="sub">Spread: push</span>'
            else:
                cov = r["home"] if marg > sp else r["away"]
                ats = f'<span class="sub">Spread: {e(cov)} covered</span>'
        vp_ = r.get("v_prob")
        flag = ""
        if vp_ is not None and not pd.isna(vp_):
            gap = (r["p_home"] - vp_) * 100
            if abs(gap) >= MARKET_FLAG_PTS:
                liked = r["home"] if gap > 0 else r["away"]
                flag = (f'<span class="flag">Model vs Vegas: {e(liked)} +{abs(gap):.0f} pts</span>')
        return f'<div class="vegas"><span class="k">Vegas</span>{a_row}{h_row}{sub}{mm}{ag}{flag}{ats}</div>'

    def vegas_line(r):
        vp = r.get("v_prob")
        if vp is None or pd.isna(vp):
            return ""
        return (f' Vegas (moneyline, no-vig): {e(r["home"])} {vp * 100:.1f}% '
                f'(model is {abs(r["p_home"] - vp) * 100:.1f} points {"higher" if r["p_home"] > vp else "lower"} on {e(r["home"])}).')

    def matchup_panel(off_team, def_team, side_off, side_def, ou, orank, du, drank):
        """One unit against the other: off_team's offense vs def_team's defense."""
        if st is None or off_team not in st.index or def_team not in st.index:
            return '<p class="note">Season stats are not available yet.</p>'
        o, d = st.loc[off_team], st.loc[def_team]
        win0 = "a" if ou > du + 1e-9 else ("h" if du > ou + 1e-9 else None)
        if win0 == "a":
            overall = f"Overall rating: {e(off_team)} offense is ahead ({ou * 100:.0f} vs {du * 100:.0f})"
        elif win0 == "h":
            overall = f"Overall rating: {e(def_team)} defense is ahead ({du * 100:.0f} vs {ou * 100:.0f})"
        else:
            overall = "Overall ratings are even"
        rows, o_adv, d_adv = [cmp_row("Overall unit rating \u2605 (model)", f"{ou * 100:.0f}{rank_badge(orank)}",
                                      f"{du * 100:.0f}{rank_badge(drank)}", win0)], 0, 0
        for label, ocol, dcol, f in MATCHUP_PAIRS:
            ro, rd = o[ocol + "_rk"], d[dcol + "_rk"]
            win = better(ro, rd, higher=False)               # lower rank number = better
            o_adv += win == "a"
            d_adv += win == "h"
            rows.append(cmp_row(label, fmt(o[ocol], f) + rank_badge(ro), fmt(d[dcol], f) + rank_badge(rd), win))
        if o_adv == d_adv:
            verdict = f"Even matchup ({o_adv}-{d_adv} by league rank)"
        elif o_adv > d_adv:
            verdict = f"Edge: {e(off_team)} offense, {o_adv} of {len(MATCHUP_PAIRS)} categories by league rank"
        else:
            verdict = f"Edge: {e(def_team)} defense, {d_adv} of {len(MATCHUP_PAIRS)} categories by league rank"
        return (f'<table class="cmp"><tr><th class="tm">{tm(off_team)} offense<br><span class="rec">{side_off}</span></th>'
                f'<th></th><th class="tm">{tm(def_team)} defense<br><span class="rec">{side_def}</span></th></tr>'
                f'{"".join(rows)}</table><p class="note">{overall}. {verdict}. Left column is the offense, right is the defense it faces. '
                f'\u2605 marks the stats behind the overall unit rating, which uses {UNIT_WINDOW_TEXT} (before kickoff); the other rows are season to date. '
                f'Field position is shown for context only and is not a model input.</p>')

    # ---- where the model and the market disagree most (unplayed games) ----
    edge_rows = []
    for r_ in pred["rows"]:
        vp_ = r_.get("v_prob")
        if r_.get("played") or vp_ is None or pd.isna(vp_):
            continue
        gap_ = (r_["p_home"] - vp_) * 100
        if abs(gap_) >= MARKET_FLAG_PTS:
            who_ = r_["home"] if gap_ > 0 else r_["away"]
            mp_ = r_["p_home"] if gap_ > 0 else 1 - r_["p_home"]
            vm_ = vp_ if gap_ > 0 else 1 - vp_
            edge_rows.append((abs(gap_), f'<div><b>{e(who_)}</b>: model {mp_ * 100:.0f}% vs market {vm_ * 100:.0f}% '
                              f'({e(r_["away"])} @ {e(r_["home"])}, {abs(gap_):.0f} pts apart)</div>'))
    edges_html = ""
    if edge_rows:
        edge_rows.sort(key=lambda t: -t[0])
        edges_html = ('<div class="edges"><h3>Where the model differs most from Vegas</h3>'
                      + "".join(h for _, h in edge_rows[:6])
                      + '<div>The market has information this model does not (line moves, late news), so treat these as '
                        'games to look at, not automatic bets.</div></div>')

    def inj_count(r):
        n = sum(1 for t in (r["away"], r["home"]) for x in (inj_reports or {}).get(t, []) if x["status"] in ("Out", "Doubtful"))
        return f' <span class="cnt">{n}</span>' if n else ""

    # ---- game cards ----
    cards, items = [], []
    for gi, r in enumerate(pred["rows"]):
        conf = max(r["p_home"], 1 - r["p_home"]) * 100
        tier = "Strong" if conf >= 65 else ("Moderate" if conf >= 57 else "Lean")
        rec = ""
        if st is not None and r["away"] in st.index and r["home"] in st.index:
            a_s, h_s = st.loc[r["away"]], st.loc[r["home"]]
            rec = (f'<span class="rec">{e(r["away"])} {e(a_s["record"])} (road {e(a_s["away_record"])}) &middot; '
                   f'{e(r["home"])} {e(h_s["record"])} (home {e(h_s["home_record"])})</span>')
        home_wins = r["p_home"] >= 0.5
        fin_html = ""
        if r.get("played"):
            fa, fh = int(r["final_a"]), int(r["final_h"])
            if fa == fh:
                fin_html = f'<span class="fin tie">FINAL {fa}&ndash;{fh} (tie)</span>'
            else:
                right = home_wins == (fh > fa)
                fin_html = (f'<span class="fin {"ok" if right else "no"}">FINAL {fa}&ndash;{fh}'
                            f' &middot; {"Pick correct" if right else "Pick wrong"}</span>')
        score_html = (f'{e(r["away"])} <span class="n{"" if home_wins else " w"}">{r["score_a"]}</span> &ndash; '
                      f'<span class="n{" w" if home_wins else ""}">{r["score_h"]}</span> {e(r["home"])}')
        vp2 = r.get("v_prob")
        flagged = bool(vp2 is not None and not pd.isna(vp2) and abs(r["p_home"] - vp2) * 100 >= MARKET_FLAG_PTS)
        if r.get("played"):
            right_ = home_wins == (int(r["final_h"]) > int(r["final_a"]))
            chip = (f'<span class="chip {"ok" if right_ else "no"}">{"&#10003;" if right_ else "&#10007;"} '
                    f'{int(r["final_a"])}&ndash;{int(r["final_h"])}</span>')
        elif flagged:
            chip = '<span class="chip flag">&ne; Vegas</span>'
        else:
            chip = f'<span class="chip">{tier}</span>'
        items.append(f'<button type="button" class="gitem" data-g="{gi}" data-played="{1 if r.get("played") else 0}">'
                     f'<span class="gi-teams">{tm(r["away"])}<span class="at">@</span>{tm(r["home"])}</span>'
                     f'<span class="gi-pick">{tm(r["winner"])} <b>{conf:.0f}%</b></span>{chip}</button>')
        players = (player_html or {}).get((r["home"], r["away"]), "")
        players_btn = '<button type="button" data-t="players">Players</button>' if players else ""
        players_tab = f'<div class="tp" data-t="players" hidden>{players}</div>' if players else ""
        cards.append(f"""
<section class="gdetail" id="g{gi}" hidden>
<div class="ghead">
  <div class="teams">{tm(r["away"], True)}<span class="at">@</span>{tm(r["home"], True)}{rec}</div>
  <div class="pick"><span class="k">Pick</span><span class="who">{tm(r["winner"], True)}</span></div>
  <div class="score"><span class="k">Predicted score</span>{score_html}{fin_html}</div>
  <div class="conf"><span class="k">Confidence</span><span class="pct">{conf:.1f}%</span><span class="tier">{tier}</span>
    <span class="bar"><span class="fill" style="width:{conf:.1f}%"></span></span></div>
  {vegas_box(r)}
</div>
<nav class="subtabs"><button type="button" data-t="over">Overview</button><button type="button" data-t="match">Matchups</button><button type="button" data-t="stats">Team stats</button><button type="button" data-t="inj">Injuries{inj_count(r)}</button>{players_btn}</nav>
<div class="tp" data-t="over">
  {weather_box(r)}
  {why_html(r)}
  {style_html(r)}
  <div class="probline">Win probability: {e(r["away"])} {100 - r["p_home"] * 100:.1f}% &middot; {e(r["home"])} {r["p_home"] * 100:.1f}%.{vegas_line(r)}</div>
  <div class="panel"><h4>Records and overall quality</h4>{records_table(r)}</div>
</div>
<div class="tp" data-t="match" hidden>
  <div class="probline">Green marks the stronger side for each stat and red the weaker (strength is by league rank). Rank badges: #1 is best in the league.</div>
  <div class="cols">
    <div class="panel"><h4>Matchup 1: {e(r["away"])} offense vs {e(r["home"])} defense</h4>{matchup_panel(r["away"], r["home"], "Away", "Home", r["a_off_unit"], r["a_off_rank"], r["h_def_unit"], r["h_def_rank"])}</div>
    <div class="panel"><h4>Matchup 2: {e(r["home"])} offense vs {e(r["away"])} defense</h4>{matchup_panel(r["home"], r["away"], "Home", "Away", r["h_off_unit"], r["h_off_rank"], r["a_def_unit"], r["a_def_rank"])}</div>
  </div>
</div>
<div class="tp" data-t="stats" hidden>
  <div class="panel"><h4>Model inputs (pre-game)</h4>{model_table(r)}</div>
</div>
<div class="tp" data-t="inj" hidden>
  <div class="probline">Practice = the last practice report of the week (DNP = did not practice; Limited; Full). Game status is the team's final designation. Out and Doubtful players are what the model counts; "Model impact" is how much each one moves the team's rating (position value x how much he plays).</div>
  <div class="cols">{injury_panel(r["away"])}{injury_panel(r["home"])}</div>
</div>
{players_tab}
</section>""")

    # ---- underdogs with a shot + parlays ----
    am_txt = lambda x: f"{x:+.0f}"

    dogs = find_underdogs(pred)
    if dogs:
        drows = "".join(
            f'<tr class="mp-row" data-g="{d["g"]}"><td class="mp-pick">{tm(d["team"], True)}</td>'
            f'<td>{"vs" if d["home"] else "@"} {tm(d["opp"])}</td><td class="num"><b>{am_txt(d["ml"])}</b></td>'
            f'<td class="num">{d["p"] * 100:.0f}%</td><td class="num">{d["mk"] * 100:.0f}%</td>'
            f'<td class="num pl-pos"><b>+{d["edge"] * 100:.0f} pts</b></td>'
            f'<td class="num {"pl-pos" if d["p"] * d["dec"] - 1 > 0 else "pl-neg"}">{(d["p"] * d["dec"] - 1) * 100:+.0f}</td>'
            f'<td>{"<b>Model pick</b>" if d["pick"] else ""}</td></tr>' for d in dogs)
        dogs_html = (
            '<div class="dogs"><h3>Underdogs with a shot</h3>'
            '<p class="sub2">Teams the betting market has as underdogs that the model rates well above their price. '
            '"Straight bet EV" is the model-estimated profit per $100 on the moneyline.</p>'
            '<div class="scroll"><table class="mp"><tr><th>Underdog</th><th>Game</th><th class="num">Odds</th><th class="num">Model</th>'
            '<th class="num">Market</th><th class="num">Edge</th><th class="num">Straight bet EV</th><th></th></tr>' + drows + '</table></div></div>')
    else:
        dogs_html = ('<div class="dogs"><h3>Underdogs with a shot</h3><p class="sub2">No underdog stands out this week: '
                     'none is both rated at least 40% by the model and clearly above the market price.</p></div>')

    def parlay_card(pl):
        legs_h = "".join(
            f'<div class="pl-leg"><div>{tm(l["team"], True)}<span class="pl-vs">{"vs" if l["home"] else "@"} {e(l["opp"])}</span></div>'
            f'<div class="pl-ml">{am_txt(l["ml"])}</div>'
            f'<div class="pl-m">Model {l["p"] * 100:.0f}% &middot; Market {l["mk"] * 100:.0f}% &middot; edge {l["edge"] * 100:+.0f} pts</div></div>'
            for l in pl["legs"])
        ev = pl["ev"] * 100
        return (f'<div class="plcard">{legs_h}<div class="pl-foot">'
                f'<div><span class="k">Parlay odds</span><b>{am_txt(dec_to_am(pl["dec"]))}</b></div>'
                f'<div><span class="k">$100 returns</span><b>${100 * pl["dec"]:,.0f}</b></div>'
                f'<div><span class="k">Model chance to hit</span><b>{pl["p"] * 100:.1f}%</b></div>'
                f'<div><span class="k">Break-even chance</span><b>{100 / pl["dec"]:.1f}%</b></div>'
                f'<div style="grid-column:1/-1"><span class="k">Model EV per $100</span>'
                f'<b class="{"pl-pos" if ev > 0 else "pl-neg"}">{ev:+.0f}</b></div></div></div>')

    tiers = build_parlays(pred)
    if tiers:
        ptxt = "".join(
            f'<div class="pltier"><h3>{e(t["label"])}</h3><p class="sub2">{e(t["blurb"])}</p>'
            f'<div class="pl-grid">{"".join(parlay_card(p) for p in t["parlays"])}</div></div>' for t in tiers)
    else:
        ptxt = ('<p class="note">No parlays to build yet: the games this week either have no betting lines in the data '
                'or are already played.</p>')
    parlays_html = (
        '<div class="toppane" id="p-parlays" hidden>'
        '<p class="pl-note"><b>Read this first.</b> These come from the model and the moneylines in the data (they can differ from your '
        'sportsbook, so check the real price). &ldquo;Model chance&rdquo; multiplies each leg&rsquo;s win probability and assumes the '
        'games are independent. Over the long run this model has been less accurate than the betting market, so its &ldquo;value&rdquo; '
        'is mostly where it disagrees with the market, not guaranteed edge. Most parlays lose. For fun, not betting advice.</p>'
        + ptxt + '</div>')

    # ---- model picks summary, most confident first ----
    mp_rows = []
    ranked_ = sorted(enumerate(pred["rows"]), key=lambda t: -max(t[1]["p_home"], 1 - t[1]["p_home"]))
    for rank_, (gi_, r) in enumerate(ranked_, 1):
        conf_ = max(r["p_home"], 1 - r["p_home"]) * 100
        tier_ = "Strong" if conf_ >= 65 else ("Moderate" if conf_ >= 57 else "Lean")
        vp_ = r.get("v_prob")
        if vp_ is None or pd.isna(vp_):
            veg_ = '<span class="mp-v">-</span>'
        else:
            mk_ = vp_ if r["winner"] == r["home"] else 1 - vp_
            far_ = abs(r["p_home"] - vp_) * 100 >= MARKET_FLAG_PTS
            veg_ = f'<span class="mp-v{" far" if far_ else ""}">Vegas {mk_ * 100:.0f}%</span>'
        res_ = ""
        if r.get("played"):
            fa_, fh_ = int(r["final_a"]), int(r["final_h"])
            if fa_ == fh_:
                res_ = f'<span class="mp-res tie">{fa_}&ndash;{fh_} tie</span>'
            else:
                ok_ = (r["winner"] == r["home"]) == (fh_ > fa_)
                res_ = f'<span class="mp-res {"ok" if ok_ else "no"}">{"&#10003;" if ok_ else "&#10007;"} {fa_}&ndash;{fh_}</span>'
        mp_rows.append(
            f'<tr class="mp-row" data-g="{gi_}"><td class="num">{rank_}</td><td class="mp-pick">{tm(r["winner"], True)}</td>'
            f'<td>{tm(r["away"])} <span class="at">@</span> {tm(r["home"])}</td>'
            f'<td class="mp-conf"><b>{conf_:.1f}%</b><span class="bar"><span class="fill" style="width:{conf_:.1f}%"></span></span></td>'
            f'<td>{tier_}</td><td class="mp-sc">{e(r["away"])} {r["score_a"]} &ndash; {r["score_h"]} {e(r["home"])}</td>'
            f'<td>{veg_}</td><td>{res_}</td></tr>')
    modelpicks_html = (
        '<div class="toppane" id="p-modelpicks" hidden>'
        '<p class="note" style="margin-top:0">Every game this week, most confident pick first. Confidence is the win probability '
        'of the picked team. Tap a row to open that game.</p>'
        '<div class="scroll"><table class="mp"><tr><th class="num">#</th><th>Pick</th><th>Game</th><th>Confidence</th><th>Tier</th>'
        '<th>Predicted score</th><th>Market</th><th>Result</th></tr>' + "".join(mp_rows) + '</table></div>' + dogs_html + '</div>')

    # ---- league table: every team's season stats with ranks ----
    if st is not None:
        order = st.sort_values("net_epa_rk")
        cols = [("Points/G", "off_ppg", "{:.1f}"), ("Pts allowed/G", "def_ppg", "{:.1f}"),
                ("Yds/G", "off_ypg", "{:.0f}"), ("Yds allowed/G", "def_ypg", "{:.0f}"),
                ("Giveaways", "off_tov", "{:.0f}"), ("Takeaways", "def_tov", "{:.0f}"),
                ("Off EPA/play", "off_epa", "{:+.3f}"), ("Def EPA/play", "def_epa", "{:+.3f}"),
                ("Off RZ TD%", "off_rz", "{:.1f}%"), ("Def RZ TD%", "def_rz", "{:.1f}%"),
                ("Expl/G", "off_expl", "{:.1f}"), ("Expl allowed/G", "def_expl", "{:.1f}"),
                ("Drive start", "off_fp", "{:.1f}"), ("Opp drive start", "def_fp", "{:.1f}")]
        head = "".join(f'<th class="num">{c[0]}</th>' for c in cols)
        body = "".join(
            f'<tr><td><b>{tm(t)}</b></td><td>{e(row["record"])}</td><td>{e(row["home_record"])}</td><td>{e(row["away_record"])}</td>'
            + "".join(f'<td class="num">{fmt(row[c], f)}{rank_badge(row[c + "_rk"])}</td>' for _, c, f in cols)
            + "</tr>" for t, row in order.iterrows())
        league = (f'<div class="scroll"><table><tr><th>Team</th><th>Record</th><th>Home</th><th>Road</th>{head}</tr>{body}</table></div>'
                  '<p class="note">Sorted by net EPA/play. Rank badges: #1 is best in the league for that stat. '
                  'Drive start is the average own yard line where a team\'s drives begin (display only, not in the model).</p>')
    else:
        league = '<p class="note">No completed games yet this season.</p>'

    # ---- Elo ----
    elo_rows = "".join(
        f'<tr><td>{tm(r["away"])}</td><td class="num">{r["a_elo"]:.0f}</td>'
        f'<td>{tm(r["home"])}</td><td class="num">{r["h_elo"]:.0f}</td></tr>' for r in pred["rows"])
    ranked = pd.Series(elo_ratings).sort_values(ascending=False)
    rank_rows = "".join(f'<tr><td class="num">{i}</td><td>{tm(t)}</td><td class="num">{v:.0f}</td></tr>'
                        for i, (t, v) in enumerate(ranked.items(), 1))

    # ---- model performance + weights ----
    mult_txt = ", ".join(f"{g} x{m:g}" for g, m in report["group_mult"].items()) or "none"
    if report["acc"] is not None:
        stats = [("Accuracy", f'{report["acc"]:.1%}'), ("Always-pick-home", f'{report["base_acc"]:.1%}'),
                 ("Log-loss", f'{report["ll"]:.3f}'),
                 ("Accuracy, learned weights only", f'{report["acc_raw"]:.1%}'),
                 ("Log-loss, learned weights only", f'{report["ll_raw"]:.3f}'),
                 ("Elo-only accuracy", f'{report["elo_acc"]:.1%}'),
                 ("Elo-only log-loss", f'{report["elo_ll"]:.3f}'),
                 *([("Accuracy without Vegas line", f'{report["novegas_acc"]:.1%}'),
                    ("Log-loss without Vegas line", f'{report["novegas_ll"]:.3f}')] if report.get("novegas_acc") else []),
                 *([("Vegas favorite accuracy", f'{report["vegas_acc"]:.1%}')] if report.get("vegas_acc") else []),
                 *([("Vegas moneyline log-loss", f'{report["vegas_ll"]:.3f}')] if report.get("vegas_ll") else []),
                 ("Margin error (avg miss, pts)", f'{report["mae_margin"]:.1f}'),
                 ("Total points error (avg miss, pts)", f'{report["mae_total"]:.1f}')]
        stats_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
                             for l, v in stats)
        perf = (f'<div class="stats">{stats_html}</div><p class="note">Walk-forward test: each season '
                f'{report.get("wf_first")}-{report.get("wf_last")} was predicted by a model trained only on earlier seasons '
                f'({report["n_test"]} games pooled). The live model is fit on all {report["n_train"]} completed games. A coin flip has log-loss 0.693. Regularization strength C = {MODEL_C:g}. Weight multipliers applied: '
                f'{e(mult_txt)}. "Learned weights only" shows the same model without those multipliers. '
                f'Guessing the average margin would miss by {report["mae_margin_base"]:.1f} pts and guessing the average total by '
                f'{report["mae_total_base"]:.1f} pts.</p>')
    else:
        perf = '<p class="note">No completed test games available yet.</p>'

    coefs = report["coefs"].reindex(report["coefs"].abs().sort_values(ascending=False).index)
    top = max(coefs.abs().max(), 1e-9)

    def mult_cell(k):
        m = report["feat_mult"].get(k, 1.0)
        return f'<td class="num">{"-" if m == 1.0 else "x%g" % m}</td>'
    coef_rows = "".join(
        f'<tr><td>{e(FEATURE_LABELS.get(k, k))}</td><td class="num">{v:+.3f}</td>{mult_cell(k)}'
        f'<td><span class="cbar {"p" if v >= 0 else "n"}" style="width:{abs(v) / top * 140:.0f}px"></span></td></tr>'
        for k, v in coefs.items())
    parts = report.get("unit_parts", {})
    sub = "".join(f'<div style="display:flex;justify-content:space-between;gap:16px;padding:2px 0 2px 12px;'
                  f'font-size:.92em;opacity:.85"><span>{e(n)}</span><span>{v:.1%}</span></div>'
                  for n, v in sorted(parts.items(), key=lambda kv: -kv[1]))
    share_rows = ""
    for g, sv in sorted(report["shares"].items(), key=lambda kv: -kv[1]):
        if g == "Unit rankings" and sub:
            share_rows += (f'<tr><td colspan="2"><details><summary style="cursor:pointer;display:flex;'
                           f'justify-content:space-between"><span>{e(g)} <small>(click for stats)</small></span>'
                           f'<span>{sv:.1%}</span></summary>{sub}</details></td></tr>')
        else:
            share_rows += f'<tr><td>{e(g)}</td><td class="num">{sv:.1%}</td></tr>'

    unit_section = ""
    if unit_table is not None and len(unit_table):
        tot_w = sum(RANK_STAT_WEIGHTS.values())
        recipe = ", ".join(f"{RANK_STAT_NAMES[k]} {w / tot_w:.0%}" for k, w in RANK_STAT_WEIGHTS.items() if w > 0)

        def unit_tbl(side, rank_col, unit_col, idx):
            heads = "".join(f'<th class="num">{e(RANK_STAT_META[k][idx])}</th>' for k in RANK_STAT_WEIGHTS)
            body = ""
            for t, x in unit_table.sort_values(rank_col).iterrows():
                cells = "".join(
                    f'<td class="num">{RANK_STAT_META[k][2].format(x[f"ur_{k}_{side}_pre"] * RANK_STAT_META[k][3])}</td>'
                    for k in RANK_STAT_WEIGHTS)
                body += (f'<tr><td><b>{tm(t)}</b></td><td class="num">{int(x[rank_col])}</td>'
                         f'<td class="num">{x[unit_col] * 100:.0f}</td>{cells}</tr>')
            return (f'<div class="scroll"><table><tr><th>Team</th><th class="num">#</th><th class="num">Rating</th>'
                    f'{heads}</tr>{body}</table></div>')

        def overall_tbl():
            ut = unit_table.copy()
            ut["ovr"] = (ut["off_unit"] + ut["def_unit"]) / 2
            ut["ovr_rank"] = ut["ovr"].rank(ascending=False, method="min")
            body = ""
            for t, x in ut.sort_values("ovr_rank").iterrows():
                body += (f'<tr><td><b>{tm(t)}</b></td><td class="num">{int(x["ovr_rank"])}</td>'
                         f'<td class="num">{x["ovr"] * 100:.0f}</td>'
                         f'<td class="num">{int(x["off_rank"])}</td><td class="num">{x["off_unit"] * 100:.0f}</td>'
                         f'<td class="num">{int(x["def_rank"])}</td><td class="num">{x["def_unit"] * 100:.0f}</td></tr>')
            return ('<div class="scroll"><table><tr><th>Team</th><th class="num">#</th><th class="num">Overall</th>'
                    '<th class="num">Off #</th><th class="num">Off rating</th>'
                    '<th class="num">Def #</th><th class="num">Def rating</th></tr>' + body + '</table></div>')

        unit_note = (f'<p class="note">Rating (0-100) blends each team\'s league percentile on: {e(recipe)}. These use '
                     f'{UNIT_WINDOW_TEXT} before kickoff. Points and yards are included but kept modest, because they '
                     'also reflect pace and game script (turnovers have their own model feature). These stats enter the model only '
                     'through this rating, so each is counted once. Overall = the average of the offense and defense ratings '
                     '(display only; the model uses the offense-vs-defense matchups). Sorted by rank; 1 is best in the league.</p>')
        unit_section = (
            '<details class="section"><summary>Offense rankings (model)</summary><div class="inner">'
            + unit_tbl("off", "off_rank", "off_unit", 0) + unit_note + '</div></details>'
            '<details class="section"><summary>Defense rankings (model)</summary><div class="inner">'
            + unit_tbl("def", "def_rank", "def_unit", 1) + unit_note + '</div></details>'
            '<details class="section"><summary>Overall rankings (offense + defense average)</summary><div class="inner">'
            + overall_tbl() + unit_note + '</div></details>')

    record_section = ""
    rec = report.get("record")
    if rec:
        n_, w_ = rec["n"], rec["w"]
        wk_rows = "".join(f'<tr><td>Week {k}</td><td class="num">{a}-{b - a}</td>'
                          f'<td class="num">{100 * a / b:.0f}%</td></tr>' for k, a, b in rec["weeks"])
        band_rows = "".join(f'<tr><td>{e(b["label"])} confidence</td><td class="num">{b["w"]}-{b["n"] - b["w"]}</td>'
                            f'<td class="num">{100 * b["w"] / b["n"]:.0f}%</td></tr>' for b in rec["bands"])
        game_rows = "".join(
            f'<tr><td>Wk {x["week"]}</td><td>{tm(x["away"])} @ {tm(x["home"])}</td><td>{tm(x["pick"])}</td>'
            f'<td class="num">{x["conf"] * 100:.0f}%</td><td class="num">{e(x["score"])}</td>'
            f'<td style="color:{"#2e9e5b" if x["ok"] else "#c0392b"}"><b>{"Correct" if x["ok"] else "Wrong"}</b></td></tr>'
            for x in rec["rows"])
        record_section = (
            f'<details class="section"><summary>Model record so far in {PREDICT_SEASON}: '
            f'{w_}-{n_ - w_} ({100 * w_ / n_:.1f}%)</summary><div class="inner">'
            f'<p class="note">Every completed {PREDICT_SEASON} game, picked with only pre-game information by a model '
            f'trained on earlier seasons. For comparison: picking by Elo alone is {rec["elo_w"]}-{n_ - rec["elo_w"]} '
            f'({100 * rec["elo_w"] / n_:.1f}%) and always picking the home team is {rec["home_w"]}-{n_ - rec["home_w"]} '
            f'({100 * rec["home_w"] / n_:.1f}%).'
            + (f' Picking the Vegas favorite (closing spread) is {rec["vegas"][0]}-{rec["vegas"][1] - rec["vegas"][0]} '
               f'({100 * rec["vegas"][0] / rec["vegas"][1]:.1f}%); it is shown only as a benchmark, the model does not use it.'
               if rec.get("vegas") else "")
            + (f' {rec["ties"]} tie(s) not counted.' if rec["ties"] else "")
            + (' REFIT_ON_ALL_COMPLETED is on, so these games were in the training data.' if REFIT_ON_ALL_COMPLETED else "")
            + '</p><div class="cols"><div class="panel"><h4>By week</h4><div class="scroll"><table>' + wk_rows + '</table></div></div>'
            '<div class="panel"><h4>By confidence</h4><div class="scroll"><table>' + band_rows + '</table></div></div></div>'
            '<details style="margin-top:12px"><summary>Every game</summary><div class="scroll"><table>'
            '<tr><th>Week</th><th>Game</th><th>Pick</th><th class="num">Confidence</th><th class="num">Final (away-home)</th><th>Result</th></tr>'
            + game_rows + '</table></div></details></div></details>')

    ab_section = ""
    ab = report.get("ablation")
    if ab:
        def ab_rows(rows, with_cost):
            out = ""
            for r in sorted(rows, key=lambda r: r["alone_vs_none"]):
                helps = r["alone_vs_none"] < -2 * r["noise"]
                out += (f'<tr><td>{e(r["name"])}</td><td class="num">{r["alone_ll"]:.4f}</td>'
                        f'<td class="num">{r["alone_acc"] * 100:.1f}%</td>'
                        f'<td class="num">{r["alone_vs_none"]:+.4f}</td>'
                        + (f'<td class="num">{r["cost_of_removing"]:+.4f}</td>' if with_cost else "")
                        + f'<td>{"clear signal" if helps else "not clearly better than guessing"}</td></tr>')
            return out
        ab_section = (
            '<details class="section"><summary>How well does each factor predict? (all seasons)</summary><div class="inner">'
            f'<p class="note">{ab["n_test"]} completed games from {ab["seasons"]}; each season is predicted by a model trained on the other seasons. Guessing the home-win rate scores log-loss '
            f'{ab["none_ll"]:.4f}; all factors together score {ab["full_ll"]:.4f} ({ab["full_acc"] * 100:.1f}% correct). '
            'Lower log-loss is better.</p>'
            '<h4>Factor groups</h4><div class="scroll"><table><tr><th>Group</th><th class="num">Alone: log-loss</th>'
            '<th class="num">Alone: accuracy</th><th class="num">vs guessing</th><th class="num">Lost if removed</th><th>Read</th></tr>'
            + ab_rows(ab["groups"], True) + '</table></div>'
            '<h4 style="margin-top:16px">Unit-rating stats, each alone</h4><div class="scroll"><table><tr><th>Stat</th>'
            '<th class="num">Log-loss</th><th class="num">Accuracy</th><th class="num">vs guessing</th><th>Read</th></tr>'
            + ab_rows(ab["stats"], False) + '</table></div>'
            '<p class="note">Learned weights only (your multipliers are ignored here). "Clear signal" means the factor beats '
            'guessing by more than twice the noise. With a few hundred games, small gaps are not reliable.</p></div></details>')

    cp = report.get("compare")
    if cp and cp.get("rows"):
        crow = "".join(f'<tr><td>{e(r["name"])}</td><td class="num">{r["acc"] * 100:.1f}%</td><td class="num">{r["ll"]:.4f}</td>'
                       f'<td class="num">{r["vs_cur"]:+.4f}</td><td class="num">{r["noise"]:.4f}</td></tr>' for r in cp["rows"])
        if cp.get("vegas"):
            crow += (f'<tr><td>Vegas (no-vig moneyline)</td><td class="num">{cp["vegas"]["acc"] * 100:.1f}%</td>'
                     f'<td class="num">{cp["vegas"]["ll"]:.4f}</td><td class="num"></td><td class="num"></td></tr>')
        ab_section += (
            '<details class="section"><summary>Is a simpler model just as good? (walk-forward comparison)</summary><div class="inner">'
            f'<p class="note">{cp["n"]} games, each season predicted by a model trained only on earlier seasons; every row uses the same games. '
            '"vs current" is the paired log-loss difference from the shipped model (negative = better); gaps smaller than about twice the noise are not reliable.</p>'
            '<div class="scroll"><table><tr><th>Set-up</th><th class="num">Accuracy</th><th class="num">Log-loss</th>'
            '<th class="num">vs current</th><th class="num">Noise</th></tr>' + crow + '</table></div></div></details>')

    warn = ('<div class="warn">No Out/Doubtful injuries were found for this week yet, so injuries '
            'are treated as zero. Injury reports usually appear midweek.</div>') if pred["no_injuries"] else ""
    stamp = datetime.now().strftime("%b %d, %Y at %I:%M %p")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Predictions - {PREDICT_SEASON} Week {pred["week"]}</title>
<style>{REPORT_CSS}{REPORT_CSS2}{logo_css}</style></head><body><div class="wrap">
<h1>NFL Predictions: {PREDICT_SEASON} Week {pred["week"]}</h1>
<p class="sub">Generated {stamp}. Pick a game on the left, then use the tabs. Confidence is the win probability of the picked team. Predicted score splits a projected game total using that win probability. Green marks the stronger side and red the weaker.</p>
{warn}
<nav class="toptabs"><button type="button" class="toptab" data-p="p-games">Games</button><button type="button" class="toptab" data-p="p-model">Record &amp; accuracy</button><button type="button" class="toptab" data-p="p-league">Teams &amp; ratings</button><button type="button" class="toptab" data-p="p-modelpicks">Model picks</button><button type="button" class="toptab" data-p="p-parlays">Parlays</button></nav>

<div class="toppane" id="p-games">
{edges_html}
<div class="app"><div class="glist">{"".join(items)}</div><div class="gview">{"".join(cards)}</div></div>
</div>

{modelpicks_html}
{parlays_html}

<div class="toppane" id="p-model" hidden><div class="secgroup">
{record_section}
<details class="section"><summary>Model performance and weights</summary><div class="inner">
{perf}
<div class="grid" style="margin-top:14px">
<div class="scroll"><table><tr><th>Feature</th><th class="num">Weight</th><th class="num">Multiplier</th><th></th></tr>{coef_rows}</table></div>
<div class="scroll"><table><tr><th>Factor</th><th class="num">Share of weight</th></tr>{share_rows}</table></div>
</div>
<p class="note">Weights are standardized logistic-regression coefficients after the manual group multipliers: positive favors the home team.
Defensive features measure yards or EPA allowed, so a negative weight is expected there. Injury differences are away minus home.</p>
</div></details>
{ab_section}
</div></div>

<div class="toppane" id="p-league" hidden><div class="secgroup">
{unit_section}
<details class="section"><summary>League team stats ({PREDICT_SEASON})</summary>
<div class="inner">{league}</div></details>
<details class="section"><summary>Elo ratings</summary><div class="inner"><div class="grid">
<div class="scroll"><table><tr><th>Away</th><th class="num">Elo</th><th>Home</th><th class="num">Elo</th></tr>{elo_rows}</table></div>
<div class="scroll"><table><tr><th class="num">#</th><th>Team</th><th class="num">Elo</th></tr>{rank_rows}</table></div>
</div><p class="note">Every team starts the season at 1500. Beating a stronger team earns more points than beating a weaker one.</p></div></details>
</div></div>

<p class="note">Predictions are statistical estimates for fun, not betting advice.</p>
<script>{REPORT_JS}</script>
</div></body></html>"""

    if os.environ.get("NFL_OUTPUT_PATH"):                   # automatic runs choose where the page goes (e.g. site/index.html)
        path = Path(os.environ["NFL_OUTPUT_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = Path(__file__).resolve().parent / REPORT_FILENAME if "__file__" in globals() else Path(REPORT_FILENAME)
    path.write_text(page, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("Loading schedules ...")
    sched = load_schedules()
    print("Loading play-by-play (this can take a minute) ...")
    pbp = load_pbp(set(sched["game_id"]))
    print("Loading players / depth charts ...")
    pos_map = load_position_map()
    depth_qb1 = load_depth_qb1()

    print("Building team-game stats and pre-game features ...")
    game_stats = build_game_stats(pbp)
    tg = build_team_games(sched, game_stats)
    tg = add_pregame_features(tg)
    choose_unit_config(tg)
    tg = add_unit_inputs(tg)
    tg, unit_grid = add_unit_ranks(tg)
    tg = add_team_hfa(tg)

    print("Building injury features ...")
    tg = add_starter_ids(tg, pbp, pos_map, depth_qb1)
    depth, depth_time = load_depth_current_qbs()
    tg = apply_depth_chart_qbs(tg, sched, depth, depth_time)
    inj = build_injury_table(tg, pbp)
    tg = tg.merge(inj, on=["team", "season", "week"], how="left")
    for c in INJ_GROUP_COLS:
        if c not in tg.columns:
            tg[c] = 0.0
    if "wr_lost" not in tg.columns:
        tg["wr_lost"] = 0.0
    tg[["qb_out", "inj_score", "wr_lost"] + INJ_GROUP_COLS] = tg[["qb_out", "inj_score", "wr_lost"] + INJ_GROUP_COLS].fillna(0)
    tg = add_qb_quality(tg, pbp)
    tg = add_continuity(tg, pbp)

    print("Building matchup features (incl. common opponents) ...")
    elo_kw = choose_elo_params(sched, build_qb_deviation(tg))
    elo, elo_ratings = compute_elo(sched, **elo_kw)
    if EPA_ELO:
        perf, _ = compute_elo(build_perf_sched(sched, tg), **elo_kw)
        elo = elo.merge(perf[["game_id", "elo_diff"]].rename(columns={"elo_diff": "elo2_diff"}), on="game_id", how="left")
    else:
        elo["elo2_diff"] = 0.0
    if PTS_RATINGS_ENABLED:
        elo = elo.merge(compute_points_ratings(sched, **choose_points_params(sched)), on="game_id", how="left")
    else:
        elo["pr_margin"], elo["pr_total"] = 0.0, 0.0
    games = build_matchups(sched, tg, elo)

    feats = model_features()
    if USE_VEGAS_FEATURE:
        nxt = games[games["home_win"].isna() & (games["season"] == PREDICT_SEASON)]
        if len(nxt):
            wk0 = nxt["week"].min()
            miss = int(nxt[nxt["week"] == wk0]["vegas_spread"].isna().sum())
            if miss:
                print(f"\n  ! {miss} game(s) in week {wk0} have no Vegas spread yet; they are treated as a pick'em, "
                      "so those predictions lean on the other factors only.")
    clean = feats + TOTAL_FEATS
    games[clean] = games[clean].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model, report = train_and_evaluate(games)
    print_elo_rankings(elo_ratings)
    report["record"] = season_record(report["record_model"], games)
    print_season_record(report["record"])
    pred = predict_upcoming(model, games, report)

    if pred is not None:
        season_stats = build_season_team_stats(sched, pbp, PREDICT_SEASON)
        unit_table = unit_grid[(unit_grid["season"] == PREDICT_SEASON)
                               & (unit_grid["week"] == pred["week"])].set_index("team")
        print("Loading team logos ...")
        logos = load_team_logos()
        print("Building WR1-vs-corner matchups ...")
        try:
            player_html = build_player_matchups(pred, tg, pbp, sched)
        except Exception as ex:                       # display-only extra; never block the report
            print(f"  ! player matchups skipped ({ex})")
            player_html = {}
        path = write_html_report(pred, report, elo_ratings, season_stats, logos, unit_table, player_html, build_injury_reports(pred))
        print(f"\nShareable report written to:\n  {path}")
        print("Send that single .html file to a friend; it opens in any web browser.")
        if OPEN_REPORT_IN_BROWSER:
            webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()