from dash import Dash, html, callback, Input, Output, State, dcc, dash_table, no_update
import dash_cytoscape as cyto
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import numpy as np
import xgboost as xgb
import colorlover

colors = {"background": "#111111", "text": "#7FDBFF"}

app = Dash(__name__)

server = app.server

# Define data paths
data_dir = Path(__file__).parent.joinpath("data")
model_dir = data_dir.joinpath("model_data")

# Load data
players = pd.read_parquet(data_dir.joinpath("players.parquet"))
plays = pd.read_parquet(data_dir.joinpath("plays.parquet"))
tackles = pd.read_parquet(data_dir.joinpath("tackles.parquet"))
games = pd.read_parquet(data_dir.joinpath("games.parquet"))

# Load models
model_xgb_2 = xgb.Booster()
model_xgb_2.load_model(model_dir.joinpath("final_xYAC_all_plays.json"))

model_xgb_pursuit = xgb.Booster()
model_xgb_pursuit.load_model(model_dir.joinpath("pursuit_train_w1-7.model"))

# Importing the model results
df_model_results = pd.read_parquet(data_dir.joinpath("mia_cin_data_for_tim.parquet"))

# define colors
color_football = ["#CBB67C", "#663831"]

# Importing the Team color information from nflfastr (https://www.nflfastr.com/reference/teams_colors_logos.html?q=colors#null)
df_teams = pd.read_csv(data_dir.joinpath("teams.csv"))

# define scale for the field
scale = 12


# Feature Engineering Functions
# dir_target_endzone calculation
def calculate_dir_target_endzone(row):
    dir = row["dir"]
    if dir < 270:
        return 90 - dir
    elif dir >= 270:
        return 450 - dir
    else:
        return np.nan


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Calculate direction with respect to the ball-carrier
def dir_wrt_bc_diff(row):
    angle = row["angle_with_bc"]
    dir_target = row["dir_target_endzone"]
    diffs = [
        abs(angle - dir_target),
        abs(angle - (dir_target - 360)),
        abs(angle - (dir_target + 360)),
    ]
    return min(diffs)


# Create features for the pursuit model
def process_data_pursuit(
    df_frame: pd.DataFrame,
    df_play: pd.DataFrame,
    df_players: pd.DataFrame,
):
    # Select needed columns from the plays DataFrame
    columns_to_select = [
        "gameId",
        "playId",
        "ballCarrierId",
        "yardsToGo",
        "possessionTeam",
        "defensiveTeam",
        "absoluteYardlineNumber",
        "playResult",
        "play_type",
    ]
    df_play_to_join = df_play[columns_to_select]

    # Select required columns from players and merge with weekly_data
    df_players_selected = df_players[["nflId", "position", "is_off", "mass_in_slugs"]]
    df_tracking_weekly = pd.merge(df_frame, df_players_selected, on="nflId", how="left")

    # Merge with tackles
    df_tracking_weekly = pd.merge(
        df_tracking_weekly, tackles, on=["gameId", "playId", "nflId"], how="left"
    )

    # Merge with play_df_to_join
    df_tracking_weekly = pd.merge(
        df_tracking_weekly, df_play_to_join, on=["gameId", "playId"], how="left"
    )

    # Create new columns
    df_tracking_weekly["target"] = np.where(
        (df_tracking_weekly["tackle"] == 1)
        | (df_tracking_weekly["assist"] == 1)
        | (df_tracking_weekly["pff_missedTackle"] == 1),
        1,
        0,
    )
    df_tracking_weekly["los"] = np.where(
        df_tracking_weekly["playDirection"] == "left",
        120 - df_tracking_weekly["absoluteYardlineNumber"],
        df_tracking_weekly["absoluteYardlineNumber"],
    )

    df_tracking_weekly["sx"] = df_tracking_weekly["s"] * np.sin(
        df_tracking_weekly["dir"] * np.pi / 180
    )
    df_tracking_weekly["sy"] = df_tracking_weekly["s"] * np.cos(
        df_tracking_weekly["dir"] * np.pi / 180
    )
    df_tracking_weekly["ax"] = df_tracking_weekly.groupby(
        ["gameId", "playId", "nflId"], group_keys=False
    ).apply(lambda g: (g.sx - g.sx.shift(1)) / 0.1)
    df_tracking_weekly["ay"] = df_tracking_weekly.groupby(
        ["gameId", "playId", "nflId"], group_keys=False
    ).apply(lambda g: (g.sy - g.sy.shift(1)) / 0.1)
    df_tracking_weekly["ax"].fillna(0, inplace=True)
    df_tracking_weekly["ay"].fillna(0, inplace=True)

    df_tracking_weekly = df_tracking_weekly[
        df_tracking_weekly["displayName"] != "football"
    ]

    # Separate out the Ball Carrier
    df_ball_carrier = df_tracking_weekly[
        df_tracking_weekly["nflId"] == df_tracking_weekly["ballCarrierId"]
    ].copy()
    # Assuming df_ball_carrier_dist is your DataFrame

    # Now you can continue with the rest of your operations

    # Select specific columns
    columns_to_select = [  # list all the columns you need here
        "gameId",
        "playId",
        "nflId",
        "frameId",
        "club",
        "possessionTeam",
        "target",
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "sx",
        "sy",
        "ax",
        "ay",
    ]

    df_ball_carrier = df_ball_carrier[columns_to_select]

    df_ball_carrier_dist = pd.merge(
        df_tracking_weekly,
        df_ball_carrier,
        how="left",
        on=["gameId", "playId", "frameId"],
        suffixes=("", "_bc"),
    )

    # Filter out rows where ball carrier NFL ID matches with the tracking NFL ID
    df_ball_carrier_dist = df_ball_carrier_dist[
        df_ball_carrier_dist["nflId_bc"] != df_ball_carrier_dist["nflId"]
    ]

    df_ball_carrier_dist["p_bc_x"] = df_ball_carrier_dist.x - df_ball_carrier_dist.x_bc
    df_ball_carrier_dist["p_bc_y"] = df_ball_carrier_dist.y - df_ball_carrier_dist.y_bc
    df_ball_carrier_dist["p_bc_d"] = euclidean_distance(
        df_ball_carrier_dist.x,
        df_ball_carrier_dist.y,
        df_ball_carrier_dist.x_bc,
        df_ball_carrier_dist.y_bc,
    )

    df_ball_carrier_dist["p_bc_t"] = (
        np.arctan2(
            df_ball_carrier_dist.x_bc - df_ball_carrier_dist.x,
            df_ball_carrier_dist.y_bc - df_ball_carrier_dist.y,
        )
        * (180 / np.pi)
        + 360
    ) % 360

    df_ball_carrier_dist["p_bc_sx"] = (
        df_ball_carrier_dist.sx - df_ball_carrier_dist.sx_bc
    )
    df_ball_carrier_dist["p_bc_sy"] = (
        df_ball_carrier_dist.sy - df_ball_carrier_dist.sy_bc
    )
    df_ball_carrier_dist["p_bc_s_rel"] = euclidean_distance(
        df_ball_carrier_dist.sx,
        df_ball_carrier_dist.sy,
        df_ball_carrier_dist.sx_bc,
        df_ball_carrier_dist.sy_bc,
    )
    df_ball_carrier_dist["p_bc_s_t"] = (
        np.arctan2(
            df_ball_carrier_dist.sx - df_ball_carrier_dist.sx_bc,
            df_ball_carrier_dist.sy - df_ball_carrier_dist.sy_bc,
        )
        * (180 / np.pi)
        + 360
    ) % 360
    df_ball_carrier_dist["p_bc_s_rel_tt"] = df_ball_carrier_dist.p_bc_s_rel * np.cos(
        abs(
            abs(abs(df_ball_carrier_dist.p_bc_s_t - df_ball_carrier_dist.p_bc_t) - 180)
            - 180
        )
        * (np.pi / 180)
    )

    df_ball_carrier_dist["p_bc_ax"] = (
        df_ball_carrier_dist.ax - df_ball_carrier_dist.ax_bc
    )
    df_ball_carrier_dist["p_bc_ay"] = (
        df_ball_carrier_dist.ay - df_ball_carrier_dist.ay_bc
    )
    df_ball_carrier_dist["p_bc_a_rel"] = euclidean_distance(
        df_ball_carrier_dist.ax,
        df_ball_carrier_dist.ay,
        df_ball_carrier_dist.ax_bc,
        df_ball_carrier_dist.ay_bc,
    )
    df_ball_carrier_dist["p_bc_a_t"] = (
        np.arctan2(
            df_ball_carrier_dist.ax - df_ball_carrier_dist.ax_bc,
            df_ball_carrier_dist.ay - df_ball_carrier_dist.ay_bc,
        )
        * (180 / np.pi)
        + 360
    ) % 360
    df_ball_carrier_dist["p_bc_a_rel_tt"] = df_ball_carrier_dist.p_bc_a_rel * np.cos(
        abs(
            abs(abs(df_ball_carrier_dist.p_bc_a_t - df_ball_carrier_dist.p_bc_t) - 180)
            - 180
        )
        * (np.pi / 180)
    )

    df_ball_carrier_dist["tm_grp"] = np.where(
        df_ball_carrier_dist["club"] == df_ball_carrier_dist["possessionTeam"], "O", "D"
    )

    # Rank players by distance within each group
    df_ball_carrier_dist["player_dist_rank"] = (
        df_ball_carrier_dist.groupby(["gameId", "playId", "frameId", "tm_grp"])[
            "p_bc_d"
        ]
        .rank(method="first")
        .astype(int)
    )

    # Flatten the MultiIndex in columns and create new column names
    # Simplified Pivot
    df_ball_carrier_dist_pivot = df_ball_carrier_dist[
        (df_ball_carrier_dist.tm_grp == "D")
        & (df_ball_carrier_dist.player_dist_rank.isin([1, 2, 3]))
    ].pivot_table(
        index=["gameId", "playId", "frameId"],
        columns=["tm_grp", "player_dist_rank"],
        values=["p_bc_s_rel", "p_bc_s_rel_tt"],  # need to add correct values to this
        aggfunc="first",
    )

    df_temp_dist_pivot = df_ball_carrier_dist.pivot_table(
        index=["gameId", "playId", "frameId"],
        columns=["tm_grp", "player_dist_rank"],
        values=["x", "y", "p_bc_d"],  # need to add correct values to this
        aggfunc="first",
    )

    # Flatten the MultiIndex in columns and create new column names
    df_ball_carrier_dist_pivot.columns = [
        "_".join(map(str, col)).strip()
        for col in df_ball_carrier_dist_pivot.columns.values
    ]
    df_temp_dist_pivot.columns = [
        "_".join(map(str, col)).strip() for col in df_temp_dist_pivot.columns.values
    ]

    # Reset the index to turn gameId, playId, and frameId into columns
    df_ball_carrier_dist_pivot = df_ball_carrier_dist_pivot.reset_index()
    df_temp_dist_pivot = df_temp_dist_pivot.reset_index()

    # Create a subset of the original DataFrame with necessary columns
    df_ball_carrier_subset = df_ball_carrier_dist.loc[
        (df_ball_carrier_dist.club != df_ball_carrier_dist.possessionTeam)
        & (df_ball_carrier_dist.frameId == max(df_ball_carrier_dist.frameId)),
        [
            "gameId",
            "playId",
            "frameId",
            "nflId",
            "target",
            "x",
            "y",
            "sx",
            "sy",
            "ax",
            "ay",
            "x_bc",
            "y_bc",
            "sx_bc",
            "sy_bc",
            "ax_bc",
            "ay_bc",
            "p_bc_x",
            "p_bc_y",
            "p_bc_d",
            "p_bc_sx",
            "p_bc_sy",
            "p_bc_s_rel",
            "p_bc_s_rel_tt",
            "p_bc_ax",
            "p_bc_ay",
            "p_bc_a_rel",
            "p_bc_a_rel_tt",
        ],
    ].drop_duplicates()

    # Merge the subset with the pivoted DataFrame
    df_ball_carrier_dist_pivot = df_ball_carrier_subset.merge(
        df_ball_carrier_dist_pivot, on=["gameId", "playId", "frameId"], how="left"
    )
    df_ball_carrier_dist_pivot = df_ball_carrier_dist_pivot.merge(
        df_temp_dist_pivot, on=["gameId", "playId", "frameId"], how="left"
    )

    bcddp = df_ball_carrier_dist_pivot.copy()  # just making a shorter name

    bcddp["def_bc_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_bc, bcddp.y_bc)

    bcddp["def_d1_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_1, bcddp.y_D_1)
    bcddp["def_d2_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_2, bcddp.y_D_2)
    bcddp["def_d3_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_3, bcddp.y_D_3)
    bcddp["def_d4_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_4, bcddp.y_D_4)
    bcddp["def_d5_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_5, bcddp.y_D_5)
    bcddp["def_d6_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_6, bcddp.y_D_6)
    bcddp["def_d7_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_7, bcddp.y_D_7)
    bcddp["def_d8_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_8, bcddp.y_D_8)
    bcddp["def_d9_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_D_9, bcddp.y_D_9)
    bcddp["def_d10_d"] = euclidean_distance(
        bcddp.x, bcddp.y, bcddp.x_D_10, bcddp.y_D_10
    )
    bcddp["def_d11_d"] = euclidean_distance(
        bcddp.x, bcddp.y, bcddp.x_D_11, bcddp.y_D_11
    )

    bcddp["def_o1_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_1, bcddp.y_O_1)
    bcddp["def_o2_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_2, bcddp.y_O_2)
    bcddp["def_o3_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_3, bcddp.y_O_3)
    bcddp["def_o4_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_4, bcddp.y_O_4)
    bcddp["def_o5_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_5, bcddp.y_O_5)
    bcddp["def_o6_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_6, bcddp.y_O_6)
    bcddp["def_o7_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_7, bcddp.y_O_7)
    bcddp["def_o8_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_8, bcddp.y_O_8)
    bcddp["def_o9_d"] = euclidean_distance(bcddp.x, bcddp.y, bcddp.x_O_9, bcddp.y_O_9)
    bcddp["def_o10_d"] = euclidean_distance(
        bcddp.x, bcddp.y, bcddp.x_O_10, bcddp.y_O_10
    )

    bcddp["def_d1_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d1_d + bcddp.p_bc_d_D_1)
    bcddp["def_d2_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d2_d + bcddp.p_bc_d_D_2)
    bcddp["def_d3_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d3_d + bcddp.p_bc_d_D_3)
    bcddp["def_d4_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d4_d + bcddp.p_bc_d_D_4)
    bcddp["def_d5_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d5_d + bcddp.p_bc_d_D_5)
    bcddp["def_d6_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d6_d + bcddp.p_bc_d_D_6)
    bcddp["def_d7_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d7_d + bcddp.p_bc_d_D_7)
    bcddp["def_d8_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d8_d + bcddp.p_bc_d_D_8)
    bcddp["def_d9_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d9_d + bcddp.p_bc_d_D_9)
    bcddp["def_d10_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d10_d + bcddp.p_bc_d_D_10)
    bcddp["def_d11_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_d11_d + bcddp.p_bc_d_D_11)

    bcddp["def_o1_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o1_d + bcddp.p_bc_d_O_1)
    bcddp["def_o2_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o2_d + bcddp.p_bc_d_O_2)
    bcddp["def_o3_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o3_d + bcddp.p_bc_d_O_3)
    bcddp["def_o4_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o4_d + bcddp.p_bc_d_O_4)
    bcddp["def_o5_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o5_d + bcddp.p_bc_d_O_5)
    bcddp["def_o6_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o6_d + bcddp.p_bc_d_O_6)
    bcddp["def_o7_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o7_d + bcddp.p_bc_d_O_7)
    bcddp["def_o8_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o8_d + bcddp.p_bc_d_O_8)
    bcddp["def_o9_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o9_d + bcddp.p_bc_d_O_9)
    bcddp["def_o10_bc_ratio"] = bcddp.def_bc_d / (bcddp.def_o10_d + bcddp.p_bc_d_O_10)

    bcddp["def_d_bc_ratio_sum"] = bcddp[
        ["def_d" + str(i) + "_bc_ratio" for i in np.arange(11) + 1]
    ].sum(axis=1)
    bcddp["def_o_bc_ratio_sum"] = bcddp[
        ["def_o" + str(i) + "_bc_ratio" for i in np.arange(10) + 1]
    ].sum(axis=1)

    # columns to keep
    non_feats = ["gameId", "playId", "frameId", "nflId", "target"]
    feats = [
        "x",
        "y",
        "sx",
        "sy",
        "ax",
        "ay",
        "x_bc",
        "y_bc",
        "sx_bc",
        "sy_bc",
        "ax_bc",
        "ay_bc",
        "p_bc_x",
        "p_bc_y",
        "p_bc_d",
        "p_bc_sx",
        "p_bc_sy",
        "p_bc_s_rel",
        "p_bc_s_rel_tt",
        "p_bc_ax",
        "p_bc_ay",
        "p_bc_a_rel",
        "p_bc_a_rel_tt",
        "p_bc_d_D_1",
        "p_bc_d_D_2",
        "p_bc_d_D_3",
        "p_bc_s_rel_D_1",
        "p_bc_s_rel_D_2",
        "p_bc_s_rel_D_3",
        "p_bc_s_rel_tt_D_1",
        "p_bc_s_rel_tt_D_2",
        "p_bc_s_rel_tt_D_3",
        "def_d_bc_ratio_sum",
        "def_o_bc_ratio_sum",
    ]

    return bcddp[non_feats + feats]


# Create the features for the xYAC model
def process_data_xyac(
    df_new_frame: pd.DataFrame, df_play: pd.DataFrame, df_players: pd.DataFrame
):
    columns_to_select = [
        "gameId",
        "playId",
        "ballCarrierId",
        "yardsToGo",
        "possessionTeam",
        "defensiveTeam",
        "defendersInTheBox",
        "absoluteYardlineNumber",
        "playResult",
        "play_type",
    ]
    df_play_to_join = df_play[columns_to_select]

    # Select required columns from players and merge with weekly_data
    df_players_selected = df_players[["nflId", "position", "is_off", "mass_in_slugs"]]
    df_tracking_weekly = pd.merge(
        df_new_frame, df_players_selected, on="nflId", how="left"
    )

    # Merge with tackles
    df_tracking_weekly = pd.merge(
        df_tracking_weekly, tackles, on=["gameId", "playId", "nflId"], how="left"
    )

    # Replace NA values in specific columns with 0
    for col in ["tackle", "assist", "forcedFumble", "pff_missedTackle"]:
        df_tracking_weekly[col].fillna(0, inplace=True)

    # Merge with play_df_to_join
    df_tracking_weekly = pd.merge(
        df_tracking_weekly, df_play_to_join, on=["gameId", "playId"], how="left"
    )

    # Create new columns
    df_tracking_weekly["isBallCarrier"] = np.where(
        df_tracking_weekly["nflId"] == df_tracking_weekly["ballCarrierId"], 1, 0
    )
    df_tracking_weekly["is_tackle"] = np.where(
        (df_tracking_weekly["tackle"] == 1) | (df_tracking_weekly["assist"] == 1), 1, 0
    )
    df_tracking_weekly["los"] = np.where(
        df_tracking_weekly["playDirection"] == "left",
        120 - df_tracking_weekly["absoluteYardlineNumber"],
        df_tracking_weekly["absoluteYardlineNumber"],
    )
    df_tracking_weekly["dist_from_los"] = (
        df_tracking_weekly["x"] - df_tracking_weekly["los"]
    )
    df_tracking_weekly["dist_from_endzone"] = 110 - df_tracking_weekly["x"]
    df_tracking_weekly["adj_y"] = df_tracking_weekly["y"] - (160 / 6)
    df_tracking_weekly["adj_x"] = 110 - df_tracking_weekly["x"]
    df_tracking_weekly["dist_to_near_sideline"] = np.minimum(
        np.abs(df_tracking_weekly["y"] - 53), np.abs(df_tracking_weekly["y"] - 0)
    )
    df_tracking_weekly["dist_to_first"] = np.maximum(
        df_tracking_weekly["yardsToGo"] - df_tracking_weekly["dist_from_los"], 0
    )

    df_tracking_weekly["dir_target_endzone"] = df_tracking_weekly.apply(
        calculate_dir_target_endzone, axis=1
    )

    df_tracking_weekly = df_tracking_weekly[
        df_tracking_weekly["displayName"] != "football"
    ]

    # Separate out the Ball Carrier
    df_ball_carrier = df_tracking_weekly[
        df_tracking_weekly["isBallCarrier"] == 1
    ].copy()

    # Now you can continue with the rest of your operations

    df_ball_carrier["x_end"] = df_ball_carrier["playResult"] + df_ball_carrier["los"]
    df_ball_carrier["ydsToGo"] = round(
        df_ball_carrier["x_end"] - df_ball_carrier["x"], 2
    )

    # Select specific columns
    columns_to_select = [  # list all the columns you need here
        "gameId",
        "playId",
        "nflId",
        "frameId",
        "play_type",
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "los",
        "playResult",
        "second",
        "dist_from_los",
        "dist_from_endzone",
        "dist_to_near_sideline",
        "dist_to_first",
        "defendersInTheBox",
        "adj_x",
        "adj_y",
        "dir_target_endzone",
        "yardsToGo",
        "x_end",
        "ydsToGo",
        "isBallCarrier",
        "displayName",
        "event",
    ]

    df_ball_carrier = df_ball_carrier[columns_to_select]

    # Complete the renaming of columns
    renamed_columns = {
        "nflId": "bc_nflId",
        "x": "bc_x",
        "y": "bc_y",
        "dir": "bc_dir",
        "dis": "bc_dis",
        "o": "bc_o",
        "a": "bc_a",
        "s": "bc_s",
        "adj_x": "bc_adj_x",
        "adj_y": "bc_adj_y",
        "dist_to_first": "bc_dist_to_first",
        "dir_target_endzone": "bc_dir_target_endzone",
        "dist_from_endzone": "bc_dist_from_endzone",
        "dist_from_los": "bc_dist_from_los",
        "dist_to_first": "bc_dist_to_first",
        "position": "bc_position",
        "displayName": "bc_displayName",
        "isBallCarrier": "bc_isBC",
    }
    df_ball_carrier.rename(columns=renamed_columns, inplace=True)

    # Merge ball_carrier with tracking_df
    df_ball_carrier_dist = pd.merge(
        df_ball_carrier,
        df_tracking_weekly,
        on=["gameId", "playId", "frameId"],
        how="left",
    )

    # Filter out rows where ball carrier NFL ID matches with the tracking NFL ID
    df_ball_carrier_dist = df_ball_carrier_dist[
        df_ball_carrier_dist["bc_nflId"] != df_ball_carrier_dist["nflId"]
    ]

    # Calculate distance to ball carrier and other metrics
    df_ball_carrier_dist["distance_to_bc"] = euclidean_distance(
        df_ball_carrier_dist["bc_x"],
        df_ball_carrier_dist["bc_y"],
        df_ball_carrier_dist["x"],
        df_ball_carrier_dist["y"],
    )
    df_ball_carrier_dist["adj_x_change"] = (
        df_ball_carrier_dist["bc_adj_x"] - df_ball_carrier_dist["adj_x"]
    )
    df_ball_carrier_dist["adj_y_change"] = (
        df_ball_carrier_dist["bc_adj_y"] - df_ball_carrier_dist["adj_y"]
    )
    df_ball_carrier_dist["angle_with_bc"] = (
        np.arctan2(
            df_ball_carrier_dist["adj_y_change"], -df_ball_carrier_dist["adj_x_change"]
        )
        * 180
        / np.pi
    )

    df_ball_carrier_dist["dir_wrt_bc_diff"] = df_ball_carrier_dist.apply(
        dir_wrt_bc_diff, axis=1
    )

    df_ball_carrier_dist["tm_grp"] = np.where(
        df_ball_carrier_dist["is_off"] == 1, "off", "def"
    )

    # Rank players by distance within each group
    df_ball_carrier_dist["player_dist_rank"] = (
        df_ball_carrier_dist.groupby(["gameId", "playId", "frameId", "tm_grp"])[
            "distance_to_bc"
        ]
        .rank(method="first")
        .astype(int)
    )

    # Flatten the MultiIndex in columns and create new column names
    # Simplified Pivot
    df_ball_carrier_dist_pivot = df_ball_carrier_dist.pivot_table(
        index=["gameId", "playId", "frameId"],
        columns=["tm_grp", "player_dist_rank"],
        values=[
            "nflId",
            "dir_target_endzone",
            "distance_to_bc",
            "adj_x_change",
            "adj_y_change",
            "angle_with_bc",
            "dir_wrt_bc_diff",
            "adj_x",
            "adj_y",
            "s",
            "dis",
        ],  # need to add correct values to this
        aggfunc="first",
    )

    # Flatten the MultiIndex in columns and create new column names
    df_ball_carrier_dist_pivot.columns = [
        "_".join(map(str, col)).strip()
        for col in df_ball_carrier_dist_pivot.columns.values
    ]

    # Adjusted logic for renaming pivoted columns
    new_column_names = {}
    for col in df_ball_carrier_dist_pivot.columns:
        parts = col.split("_")
        # Identify 'tm_grp' (off or def) and 'player_dist_rank' in the column name
        tm_grp = [part for part in parts if part in ["off", "def"]]
        player_dist_rank = [part for part in parts if part.isdigit()]

        if tm_grp and player_dist_rank:
            # Extract remaining parts (attribute name)
            attribute_parts = [
                part for part in parts if part not in tm_grp + player_dist_rank
            ]
            attribute_name = "_".join(attribute_parts)

            # Rearrange to get 'tm_grp_playerRank_attribute'
            new_col_name = f"{tm_grp[0]}_{player_dist_rank[0]}_{attribute_name}"
            new_column_names[col] = new_col_name
        else:
            new_column_names[col] = col

    # Apply the new column names
    df_ball_carrier_dist_pivot.rename(columns=new_column_names, inplace=True)

    # Reset the index to turn gameId, playId, and frameId into columns
    df_ball_carrier_dist_pivot = df_ball_carrier_dist_pivot.reset_index()

    # Create a subset of the original DataFrame with necessary columns
    df_ball_carrier_subset = df_ball_carrier[
        [
            "gameId",
            "playId",
            "frameId",
            "bc_nflId",
            "bc_x",
            "bc_y",
            "bc_s",
            "bc_a",
            "bc_o",
            "bc_dir",
            "bc_dis",
            "bc_dist_from_los",
            "bc_dist_from_endzone",
            "bc_dist_to_first",
            "bc_adj_x",
            "bc_adj_y",
            "bc_dir_target_endzone",
            "dist_to_near_sideline",
            "defendersInTheBox",
        ]
    ].drop_duplicates()

    # Merge the subset with the pivoted DataFrame
    df_ball_carrier_dist_pivot = df_ball_carrier_dist_pivot.merge(
        df_ball_carrier_subset, on=["gameId", "playId", "frameId"], how="left"
    )

    return df_ball_carrier_dist_pivot


# Plot the different parts of the field
def plot_field(line_of_scrimmage, first_down_line):
    field_data = []

    field_data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    field_data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[53.5 - 5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # plot line of scrimmage
    field_data.append(
        go.Scatter(
            x=[line_of_scrimmage, line_of_scrimmage],
            y=[0, 53.5],
            line_dash="dash",
            line_color="blue",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # plot first down line
    field_data.append(
        go.Scatter(
            x=[first_down_line, first_down_line],
            y=[0, 53.5],
            line_dash="dash",
            line_color="red",
            showlegend=False,
            hoverinfo="none",
        )
    )

    return field_data


# Create the Elements dictionary for the Dash Cytoscape Graph
def frame_data(df_frame, ball_carrier_id, tackle_id, n_clicks=0):
    df_frame.loc[df_frame["nflId"] == ball_carrier_id, "club"] = "bc"
    df_frame.loc[df_frame["nflId"].isin(tackle_id), "club"] = "tackle"
    data = (
        df_frame[["nflId", "displayName", "jerseyNumber", "x", "y", "club"]]
        .apply(
            lambda x: {
                "data": {
                    "id": "{}-{}".format(x["nflId"], n_clicks),
                    "label": x["jerseyNumber"],
                    "classes": x["club"],
                },
                "position": {"x": (x["x"]) * scale, "y": (53.3 - x["y"]) * scale},
                "grabbable": False if x["club"] == "football" else True,
            },
            axis=1,
        )
        .tolist()
    )
    return data

#define the background color bins for the data table
def discrete_background_color_bins(df, n_bins=5, columns='all'):
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Greens'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
    return styles


# Create the data frame for the adjusted coordinates from the Dash Cytoscape Graph
def create_data(elements, df_frame, scale=15):
    df = pd.json_normalize(elements)
    df.rename(
        columns={
            "data.id": "nflId",
            "data.label": "jerseyNumber",
            "data.classes": "club",
            "position.x": "x",
            "position.y": "y",
        },
        inplace=True,
    )
    df["nflId"] = df["nflId"].str.split("-").str[0].astype("float64")
    df["x"] = df["x"].apply(lambda x: round((x / scale), 2))
    df["y"] = df["y"].apply(lambda x: round(53.3 - (x / scale), 2))

    old_columns = [
        "gameId",
        "playId",
        "nflId",
        "displayName",
        "frameId",
        "time",
        "jerseyNumber",
        "club",
        "playDirection",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "event",
        "second",
    ]
    new_colums = ["nflId", "x", "y"]
    df = pd.merge(df_frame[old_columns], df[new_colums], on="nflId", how="left")

    return df


# Getting all the information together for the application
# Select the week 3
gameId = 2022092900

df_game_plays = plays[plays.gameId == gameId]
initial_playId = df_game_plays["playId"].sort_values().unique()[0]

df_initial_play = plays[(plays.gameId == gameId) & (plays.playId == initial_playId)]

df_tracking_initial_play = df_model_results[(df_model_results.playId == initial_playId)]
initial_frames = df_tracking_initial_play["frameId"].unique()
initial_frameId = df_tracking_initial_play["frameId"].unique()[0]

df_initial_frame = df_tracking_initial_play[df_tracking_initial_play["frameId"] == initial_frameId]

df_model_results_frame = df_model_results[
    (df_model_results["gameId"] == gameId)
    & (df_model_results["playId"] == initial_playId)
    & (df_model_results["frameId"] == initial_frameId)
].copy()
df_model_results_frame["tackle_prob_new"] = df_model_results_frame["tackle_prob"]
df_model_results_frame["tackle_prob_delta"] = (
    df_model_results_frame["tackle_prob_new"] - df_model_results_frame["tackle_prob"]
)
df_model_results_frame["xYAC_new"] = df_model_results_frame["xYAC"]
df_model_results_frame["xYAC_delta"] = (
    df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
)

df_model_results_frame = df_model_results_frame[
    df_model_results_frame["club"] != "football"
]

df_results_bc = df_model_results_frame.loc[
    df_model_results_frame["isBallCarrier"] == 1,
    ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
].copy()
df_results_bc = df_results_bc.round(2)
df_results_bc.rename(
        columns = {
            "displayName": "Player",
            "jerseyNumber": "Jersey Number",
            "club": "Team",
            "xYAC": "Expected Yards",
            "xYAC_new": "New Expected Yards",
            "xYAC_delta": "Delta Expected Yards",
        },
    )

df_results_tackle = df_model_results_frame.loc[
    ~df_model_results_frame["tackle_prob"].isna(),
    [
        "displayName",
        "jerseyNumber",
        "club",
        "is_tackle",
        "tackle_prob",
        "tackle_prob_new",
        "tackle_prob_delta",
    ],
].copy()
df_results_tackle = df_results_tackle.round(2)
df_results_tackle.rename(
    columns = {
        "displayName": "Player",
        "jerseyNumber": "Jersey Number",
        "club": "Team",
        "is_tackle": "Tackle",
        "tackle_prob": "Tackle Probability",
        "tackle_prob_new": "New Tackle Probability",
        "tackle_prob_delta": "Delta Tackle Probability",
    }
)

# Make sure that the play is displayed in the correct direction
if df_tracking_initial_play.playDirection.values[0] == "right":
    line_of_scrimmage = df_initial_play.absoluteYardlineNumber.values[0]
    first_down_line = (
        df_initial_play.absoluteYardlineNumber.values[0] + df_initial_play.yardsToGo.values[0]
    )
else:
    line_of_scrimmage = 120 - df_initial_play.absoluteYardlineNumber.values[0]
    first_down_line = (
        120 - df_initial_play.absoluteYardlineNumber.values[0] + df_initial_play.yardsToGo.values[0]
    )

teams = df_game_plays["possessionTeam"].unique()

# Define the ids for the inital tacklers and ball cariiers
ball_carrier_id = df_model_results_frame["ballCarrierId"].values[0]
tackle_id = df_model_results_frame.loc[df_model_results_frame['is_tackle'] == 1, 'nflId'].values
data = frame_data(df_frame=df_initial_frame, tackle_id=tackle_id, ball_carrier_id=ball_carrier_id)

# Create interactice figure
layout_interactive = go.Layout(
    autosize=True,
    width=120 * scale,
    height=53.3 * scale,
    xaxis=dict(
        range=[0, 120],
        autorange=False,
        tickmode="array",
        tickvals=np.arange(10, 111, 5).tolist(),
        showticklabels=False,
    ),
    yaxis=dict(range=[0, 53.3], autorange=False, showgrid=False, showticklabels=False),
    plot_bgcolor="#00B140",
    margin=dict(l=0, r=0, t=0, b=0),
)


fig_interactive = go.Figure(
    data=plot_field(
        line_of_scrimmage=line_of_scrimmage, first_down_line=first_down_line
    ),
    layout=layout_interactive,
)

# Define the stylesheet for the Dash Cytoscape Graph where the classes for home team, away team, football, ball carrier and tackler are defined. 
stylesheet = [
    {"selector": "node", "style": {"content": "data(label)"}},
    {
        "selector": '[classes = "football"]',
        "style": {
            "width": 15,
            "height": 10,
            "background-color": color_football[0],
            "border-color": color_football[1],
            "border-width": 2,
        },
    },
    {
        "selector": '[classes = "{}"]'.format(teams[0]),
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": df_teams.loc[
                df_teams["team_abbr"] == teams[0], "team_color"
            ].values[0],
            "border-color": df_teams.loc[
                df_teams["team_abbr"] == teams[0], "team_color2"
            ].values[0],
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "{}"]'.format(teams[1]),
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": df_teams.loc[
                df_teams["team_abbr"] == teams[1], "team_color"
            ].values[0],
            "border-color": df_teams.loc[
                df_teams["team_abbr"] == teams[1], "team_color2"
            ].values[0],
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "bc"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "red",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "bc"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "red",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "tackle"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "#ffcc00",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
]

# Define the layout of the dashboard
app.layout = html.Div(
    [
        # Explanation of the app
        html.Div(
            [
                dcc.Markdown('''
                    **How to use the app:**
                    * Select the play you want to look at from the dropdown menu.
                    * Use the slider to move through the frames of the play.
                    * Drag the players to the positions where you want them.
                        * The red dot with the white border is the ball carrier.
                        * The yellow dot with the white border is the tackler.
                        * The orange dots with the black border are Cincinnati Bengals players.
                        * The teal dots with the orange border are Miami Dolphins players.
                    * **Score Modified Play** - Click this button to recalculate the **Expected Yards** and **Tackle Probability** metrics for the new positions.
                    * **Start/Stop Play** - Click this button to start and stop the animiation of the play in a loop (The animation also stops by selecting a new play).
                    * In the tables down below the results for the Metrics Expected Yards and Tackle Probability are shown for the Ball Carrier and the Possible Tacklers.
                        * **Expected Yards** - The expected yards for the ball carrier on the play. This is calculated using the xYAC model.
                        * **Tackle Probability** - The probability that the player will make the tackle on the ball carrier. This is calculated using the pursuit model.
                    * After scoring the modified play, the table shows the results for the ball carrier and the possible tacklers for the modified play and the difference from the original play.
                    
                    Note: 
                    * The plays start with the second available frame. 
                    * For run plays the Expected Yards metric is available from the moment the ball is snapped.
                ''')
            ]
        ),
        html.Br(),
        html.Div(
            [
                # The dropdown menu to select the play
                dcc.Dropdown(
                    options=df_game_plays[["playId", "playDescription"]]
                    .sort_values("playId")
                    .apply(
                        lambda x: {
                            "label": "{} - {}".format(
                                x["playId"], x["playDescription"]
                            ),
                            "value": x["playId"],
                        },
                        axis=1,
                    )
                    .values.tolist(),
                    value=df_game_plays["playId"].sort_values().unique()[0],
                    clearable=False,
                    id="play-selection",
                ),
                html.Br(),
                # The buttons to start and stop the play and the slider to move through the frames
                html.Div(
                    [
                        html.Button("Start/Stop Play", id="btn-play", style={"margin-left": "10px"}),
                    ]
                ),
                html.Br(),
                dcc.Slider(
                    min=2,
                    max=len(initial_frames),
                    step=1,
                    value=2,
                    marks=dict(
                        zip(
                            [x for x in range(2, len(initial_frames) + 1)],
                            [{"label": "{}".format(int(x))} for x in initial_frames[1:]],
                        )
                    ),
                    id="slider-frame",
                ),
            ],
            style={"width": "75%", "display": "inline-block"},
        ),
        html.Br(),
        # The implementation of the Scorebug to get the information of the play
        html.Img(
            id='home-logo',
            src=df_teams[df_teams['team_abbr'] == 'CIN']['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10%"}
        ),
        html.H1(
            id="score",
            children="""{} - {}""".format(df_initial_play['preSnapHomeScore'].values[0], df_initial_play['preSnapVisitorScore'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "10px"},
        ),
        html.Img(
            id='away-logo',
            src=df_teams[df_teams['team_abbr'] == 'MIA']['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10px"}
        ),
        html.H1(
            id="down-distance",
            children="""{} & {}""".format(df_initial_play['down'].values[0], df_initial_play['yardsToGo'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="time",
            children="""{}""".format(df_initial_play['gameClock'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="quarter",
            # children="""{}th Quarter""".format(df_initial_play['quarter'].values[0]),
            children='4th Quarter' if df_initial_play['quarter'].values[0] == 4 else '1st Quarter' if df_initial_play['quarter'].values[0] == 1 else '2nd Quarter' if df_initial_play['quarter'].values[0] == 2 else '3rd Quarter',
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="possession-name",
            children="""Possession Team: """,
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.Img(
            id='possession',
            src=df_teams[df_teams['team_abbr'] == df_initial_play['possessionTeam'].values[0]]['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10px"}
        ),
        html.Div(
            [
                # The field plot and the Dash Cytoscape Graph on top of it. Absolute measurements needed to make sure the two are aligned.
                html.Div(
                    [
                        dcc.Graph(
                            id="plot-field",
                            figure=fig_interactive,
                        ),
                    ],
                    style={
                        "position": "absolute",
                        "width": "{}px".format(120 * scale),
                        "height": "{}px".format(53.3 * scale),
                    },
                ),
                html.Div(
                    [
                        cyto.Cytoscape(
                            id="cytoscape-test",
                            layout={
                                "name": "preset",
                                "fit": True,
                            },
                            style={
                                "position": "absolute",
                                "width": "{}px".format(120 * scale),
                                "height": "{}px".format(53.3 * scale),
                            },
                            elements=data,
                            zoom=1,
                            zoomingEnabled=False,
                            panningEnabled=False,
                            stylesheet=stylesheet,
                        ),
                    ]
                ),
            ]
        ),
        
        html.Div([html.Br() for x in range(36)]),
        html.Br(),
        # The button to score the modified play
        html.Div(
            [
                html.Button("Score Modified Play", id="btn-click"),
            ]
        ),
        # The data tables to show the model results
        html.Div(
            [
                html.H3("Ball Carrier Results"),
                dash_table.DataTable(
                    data=df_results_bc.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_results_bc.columns],
                    style_data_conditional=discrete_background_color_bins(df_results_bc, columns=['xYAC', 'xYAC_new', 'xYAC_delta']),
                    id="tbl-results-bc",
                ),
                html.H3("Possible Tackler Results"),
                dash_table.DataTable(
                    data=df_results_tackle.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_results_tackle.columns],
                    style_data_conditional=discrete_background_color_bins(df_results_tackle, columns=['tackle_prob', 'tackle_prob_new', 'tackle_prob_delta']),
                    id="tbl-results-tackle",
                ),
            ],
            style={"width": "60%", "float": "left", "display": "inline-block"},
        ),
        html.Br(),
        html.Div(
            [
                dcc.Interval(id="animate", interval=200, disabled=True),
            ]
        )
    ]
)


# Change the play based on the dropdown selection
@callback(
    Output("slider-frame", "max"),
    Output("slider-frame", "marks"),
    Output("slider-frame", "value"),
    Output("plot-field", "figure"),
    Output("score", "children"),
    Output("down-distance", "children"),
    Output("time", "children"),
    Output("quarter", "children"),
    Output("possession", "src"),
    Output("animate", "disabled", allow_duplicate=True),
    Input("play-selection", "value"),
    prevent_initial_call=True,
)
def change_play(playId):
    df_play = plays[(plays.gameId == gameId) & (plays.playId == playId)]

    df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()

    max = len(frames)
    marks = dict(
        zip(
            [x for x in range(2, len(frames) + 1)],
            [{"label": "{}".format(int(x))} for x in frames[1:]],
        )
    )
    value = 2

    if df_tracking_play.playDirection.values[0] == "right":
        line_of_scrimmage = df_play.absoluteYardlineNumber.values[0]
        first_down_line = (
            df_play.absoluteYardlineNumber.values[0] + df_play.yardsToGo.values[0]
        )
    else:
        line_of_scrimmage = 120 - df_play.absoluteYardlineNumber.values[0]
        first_down_line = (
            120 - df_play.absoluteYardlineNumber.values[0] + df_play.yardsToGo.values[0]
        )

    fig_interactive = go.Figure(
        data=plot_field(
            line_of_scrimmage=line_of_scrimmage, first_down_line=first_down_line
        ),
        layout=layout_interactive,
    )

    score = "{} - {}".format(df_play['preSnapHomeScore'].values[0], df_play['preSnapVisitorScore'].values[0])
    down_distance = "{} & {}".format(df_play['down'].values[0], df_play['yardsToGo'].values[0])
    time = "{}".format(df_play['gameClock'].values[0])
    quarter = '4th Quarter' if df_play['quarter'].values[0] == 4 else '1st Quarter' if df_play['quarter'].values[0] == 1 else '2nd Quarter' if df_play['quarter'].values[0] == 2 else '3rd Quarter'
    possession = df_teams[df_teams['team_abbr'] == df_play['possessionTeam'].values[0]]['team_logo_wikipedia'].values[0]

    return max, marks, value, fig_interactive, score, down_distance, time, quarter, possession, True


# Change the Dash Cytoscape Graph based on the slider selection and the play selection
@callback(
    Output("cytoscape-test", "elements", allow_duplicate=True),
    Output("btn-click", "n_clicks"),
    Output("tbl-results-bc", "data", allow_duplicate=True),
    Output("tbl-results-bc", "columns", allow_duplicate=True),
    Output("tbl-results-bc", "style_data_conditional", allow_duplicate=True),
    Output("tbl-results-tackle", "data", allow_duplicate=True),
    Output("tbl-results-tackle", "columns", allow_duplicate=True),
    Output("tbl-results-tackle", "style_data_conditional", allow_duplicate=True),
    Input("slider-frame", "value"),
    Input("play-selection", "value"),
    prevent_initial_call=True,
)
def change_frame(sliderValue, playId):
    # df_tracking_play = weekly_data[
    #     (weekly_data.gameId == gameId) & (weekly_data.playId == playId)
    # ]
    df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()
    frameId = frames[sliderValue - 1]

    # df_frame = weekly_data[
    #     (weekly_data["gameId"] == gameId)
    #     & (weekly_data["playId"] == playId)
    #     & (weekly_data["frameId"] == frameId)
    # ]
    df_frame = df_tracking_play[df_tracking_play["frameId"] == frameId]

    df_model_results_frame = df_model_results[
        (df_model_results["gameId"] == gameId)
        & (df_model_results["playId"] == playId)
        & (df_model_results["frameId"] == frameId)
    ].copy()
    df_model_results_frame["tackle_prob_new"] = df_model_results_frame["tackle_prob"]
    df_model_results_frame["tackle_prob_delta"] = (
        df_model_results_frame["tackle_prob_new"]
        - df_model_results_frame["tackle_prob"]
    )
    df_model_results_frame["xYAC_new"] = df_model_results_frame["xYAC"]
    df_model_results_frame["xYAC_delta"] = (
        df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
    )

    df_model_results_frame = df_model_results_frame[
        df_model_results_frame["club"] != "football"
    ]

    df_results_bc = df_model_results_frame.loc[
        df_model_results_frame["isBallCarrier"] == 1,
        ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
    ].copy()
    df_results_bc = df_results_bc.round(2)
    df_results_bc.rename(
        columns = {
            "displayName": "Player",
            "jerseyNumber": "Jersey Number",
            "club": "Team",
            "xYAC": "Expected Yards",
            "xYAC_new": "New Expected Yards",
            "xYAC_delta": "Delta Expected Yards",
        },
        inplace=True,
    )

    df_results_tackle = df_model_results_frame.loc[
        ~df_model_results_frame["tackle_prob"].isna(),
        [
            "displayName",
            "jerseyNumber",
            "club",
            "is_tackle",
            "tackle_prob",
            "tackle_prob_new",
            "tackle_prob_delta",
        ],
    ].copy()
    df_results_tackle = df_results_tackle.round(2)
    df_results_tackle.rename(
        columns = {
            "displayName": "Player",
            "jerseyNumber": "Jersey Number",
            "club": "Team",
            "is_tackle": "Tackle",
            "tackle_prob": "Tackle Probability",
            "tackle_prob_new": "New Tackle Probability",
            "tackle_prob_delta": "Delta Tackle Probability",
        },
        inplace=True,
    )
    df_results_tackle.sort_values(by=["New Tackle Probability"], ascending=False, inplace=True)

    style_bc = discrete_background_color_bins(df_results_bc, columns=['Expected Yards', 'New Expected Yards', 'Delta Expected Yards'])
    style_tackle = discrete_background_color_bins(df_results_tackle, columns=['Tackle Probability', 'New Tackle Probability', 'Delta Tackle Probability'])

    data_bc = df_results_bc.to_dict("records")
    columns_bc = [{"name": i, "id": i} for i in df_results_bc.columns]
    data_tackle = df_results_tackle.to_dict("records")
    columns_tackle = [{"name": i, "id": i} for i in df_results_tackle.columns]

    ball_carrier_id = df_model_results_frame["ballCarrierId"].values[0]
    tackle_id = df_model_results_frame.loc[df_model_results_frame['is_tackle'] == 1, 'nflId'].values
    data = frame_data(df_frame=df_frame, ball_carrier_id=ball_carrier_id, tackle_id=tackle_id, n_clicks=sliderValue)
    n_clicks = None

    return data, n_clicks, data_bc, columns_bc, style_bc, data_tackle, columns_tackle, style_tackle


# Callback to animate the play, updating the sliders, as long the animate input is not disabled
@app.callback(
    Output("slider-frame", "value", allow_duplicate=True),
    Input('animate', 'n_intervals'),
    State("play-selection", "value"),
    State("slider-frame", 'value'),
    prevent_initial_call=True,
)
def update_output(n, playId, sliderValue):
    df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()
    if sliderValue < len(frames):
        sliderValue += 1
    else:
        sliderValue = 2
    return sliderValue

# Callback to toggle the animation
@app.callback(
    Output("animate", "disabled", allow_duplicate=True),
    Input("btn-play", "n_clicks"),
    State("animate", "disabled"),
    prevent_initial_call=True,
)
def toggle(n, playing):
    if n:
        return not playing
    return playing

# Callback to the the Coordinates from the Dash Cytoscape Graph
@callback(
    # Output("output-coords", "children"),
    Output("tbl-results-bc", "data"),
    Output("tbl-results-bc", "columns"),
    Output("tbl-results-tackle", "data"),
    Output("tbl-results-tackle", "columns"),
    Input("btn-click", "n_clicks"),
    Input("slider-frame", "value"),
    Input("play-selection", "value"),
    State("cytoscape-test", "elements"),
    prevent_initial_call=True,
)
def get_coordinates(btn_click, sliderValue, playId, elements):
    if btn_click:
        gameId = 2022092900

        df_tracking_play = df_model_results[(df_model_results.playId == playId)]
        frames = df_tracking_play["frameId"].unique()
        frameId = frames[sliderValue - 1]

        df_frame = df_tracking_play[df_tracking_play["frameId"] == frameId]

        df_play = plays[(plays.gameId == gameId) & (plays.playId == playId)]

        df_test = create_data(elements=elements, df_frame=df_frame, scale=scale)

        df_testing_new_frame = process_data_xyac(
            df_new_frame=df_test,
            df_play=df_play,
            df_players=players,
        )
        features_list = model_xgb_2.feature_names
        results = model_xgb_2.predict(xgb.DMatrix(df_testing_new_frame[features_list]))

        gameId, playId, frameId = df_test.iloc[0][["gameId", "playId", "frameId"]]
        df_pursuit = df_tracking_play[df_tracking_play["frameId"] == (frameId - 1)]
        df_pursuit = df_pursuit[['gameId', 'playId', 'nflId', 'displayName', 'frameId', 'time','jerseyNumber', 'club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o','dir', 'event', 'second']].copy()
        df_pursuit = pd.DataFrame(pd.concat([df_pursuit, df_test], axis=0))
        df_test_bcddp = process_data_pursuit(
            df_frame=df_pursuit,
            df_play=df_play,
            df_players=players,
        )

        non_feats = ["gameId", "playId", "frameId", "nflId", "target"]
        feats = [
            "x",
            "y",
            "sx",
            "sy",
            "ax",
            "ay",
            "x_bc",
            "y_bc",
            "sx_bc",
            "sy_bc",
            "ax_bc",
            "ay_bc",
            "p_bc_x",
            "p_bc_y",
            "p_bc_d",
            "p_bc_sx",
            "p_bc_sy",
            "p_bc_s_rel",
            "p_bc_s_rel_tt",
            "p_bc_ax",
            "p_bc_ay",
            "p_bc_a_rel",
            "p_bc_a_rel_tt",
            "p_bc_d_D_1",
            "p_bc_d_D_2",
            "p_bc_d_D_3",
            "p_bc_s_rel_D_1",
            "p_bc_s_rel_D_2",
            "p_bc_s_rel_D_3",
            "p_bc_s_rel_tt_D_1",
            "p_bc_s_rel_tt_D_2",
            "p_bc_s_rel_tt_D_3",
            "def_d_bc_ratio_sum",
            "def_o_bc_ratio_sum",
        ]

        df_bcddp_results = df_test_bcddp[non_feats].copy()
        df_bcddp_results["pred_pursuit"] = model_xgb_pursuit.predict(
            xgb.DMatrix(df_test_bcddp[feats])
        )
        df_bcddp_results = df_bcddp_results.merge(
            df_test[["nflId", "displayName", "jerseyNumber", "club"]],
            on=["nflId"],
            how="left",
        )

        df_model_results_frame = df_model_results[
            (df_model_results["gameId"] == gameId)
            & (df_model_results["playId"] == playId)
            & (df_model_results["frameId"] == frameId)
        ].copy()

        df_model_results_frame = df_model_results_frame.merge(
            df_bcddp_results[["nflId", "pred_pursuit", "target"]],
            on=["nflId"],
            how="left",
        )
        df_model_results_frame.rename(
            columns={"pred_pursuit": "tackle_prob_new"}, inplace=True
        )

        df_model_results_frame["tackle_prob_delta"] = (
            df_model_results_frame["tackle_prob_new"]
            - df_model_results_frame["tackle_prob"]
        )
        df_model_results_frame["xYAC_new"] = round(results[0], 2)
        df_model_results_frame["xYAC_delta"] = (
            df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
        )

        df_model_results_frame = df_model_results_frame[
            df_model_results_frame["club"] != "football"
        ]

        df_results_bc = df_model_results_frame.loc[
            df_model_results_frame["isBallCarrier"] == 1,
            ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
        ].copy()
        df_results_bc["xYAC_new"] = df_results_bc["xYAC_new"].astype("float64")
        df_results_bc = df_results_bc.round(2)
        df_results_bc.rename(
            columns = {
                "displayName": "Player",
                "jerseyNumber": "Jersey Number",
                "club": "Team",
                "xYAC": "Expected Yards",
                "xYAC_new": "New Expected Yards",
                "xYAC_delta": "Delta Expected Yards",
            },
            inplace=True,
        )


        df_results_tackle = df_model_results_frame.loc[
            ~df_model_results_frame["tackle_prob"].isna(),
            [
                "displayName",
                "jerseyNumber",
                "club",
                "is_tackle",
                "tackle_prob",
                "tackle_prob_new",
                "tackle_prob_delta",
            ],
        ].copy()
        df_results_tackle["tackle_prob_new"] = df_results_tackle[
            "tackle_prob_new"
        ].astype("float64")
        df_results_tackle = df_results_tackle.round(2)
        df_results_tackle.rename(
            columns = {
                "displayName": "Player",
                "jerseyNumber": "Jersey Number",
                "club": "Team",
                "is_tackle": "Tackle",
                "tackle_prob": "Tackle Probability",
                "tackle_prob_new": "New Tackle Probability",
                "tackle_prob_delta": "Delta Tackle Probability",
            },
            inplace=True,
        )
        df_results_tackle.sort_values(by=["New Tackle Probability"], ascending=False, inplace=True)

        data_bc = df_results_bc.to_dict("records")
        columns_bc = [{"name": i, "id": i} for i in df_results_bc.columns]
        data_tackle = df_results_tackle.to_dict("records")
        columns_tackle = [{"name": i, "id": i} for i in df_results_tackle.columns]

        return data_bc, columns_bc, data_tackle, columns_tackle
    else:
        return no_update, no_update, no_update, no_update


if __name__ == "__main__":
    app.run(debug=True)
