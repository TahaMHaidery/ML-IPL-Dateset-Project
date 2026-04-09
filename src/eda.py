"""
eda.py  ─  Comprehensive IPL Exploratory Data Analysis
=======================================================
Visualizations:
  Basic:
    1.  Histogram         — runs distribution (deliveries)
    2.  Bar Graph         — top-10 match-winning teams
    3.  Pie Chart         — toss decision split
    4.  Scatter Plot      — toss winner vs match winner
    5.  Box-Whisker       — runs per over by team
    6.  KDE Plot          — first-innings score density
  Advanced:
    7.  Heatmap           — head-to-head win matrix
    8.  Violin Plot       — season-wise first-innings score
    9.  Pair Plot         — winner-prediction features
   10.  Stacked Bar       — win rate: bat vs field after winning toss
   11.  Bubble Chart      — venue stats (matches × win rate × avg score)
   12.  Rolling Form      — top-4 teams' 10-match rolling win rate
   13.  Correlation Matrix— feature correlations (winner model)
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
})
PALETTE = "Set2"
IPL_BLUE = "#004BA0"
IPL_GOLD = "#FFD700"

TEAM_MAP = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Rising Pune Supergiant":      "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

OUT = "outputs/plots"
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────
def load_data():
    matches    = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")

    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in matches.columns:
            matches[col] = matches[col].replace(TEAM_MAP)
    for col in ["batting_team", "bowling_team"]:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].replace(TEAM_MAP)

    matches["venue"] = (matches["venue"].astype(str)
                        .str.split(",").str[0].str.strip().str.title())
    matches["date"]  = pd.to_datetime(matches["date"], errors="coerce")
    matches["season"] = matches["season"].astype(str).str.extract(r"(\d{4})").astype(int)
    return matches, deliveries


def _savefig(name, fig=None):
    path = f"{OUT}/{name}.png"
    (fig or plt).savefig(path, bbox_inches="tight")
    plt.close("all")
    print(f"  ✅  {name}.png")
    return path


# ─────────────────────────────────────────────────────────────────
# 1. HISTOGRAM — runs per ball distribution
# ─────────────────────────────────────────────────────────────────
def plot_histogram(deliveries):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution of Runs in IPL", fontsize=15, fontweight="bold")

    # 1a — batsman runs per delivery
    ax = axes[0]
    vals = deliveries["batsman_runs"].value_counts().sort_index()
    ax.bar(vals.index, vals.values, color=sns.color_palette(PALETTE, len(vals)))
    ax.set_xlabel("Batsman Runs per Ball")
    ax.set_ylabel("Frequency")
    ax.set_title("Runs per Delivery")

    # 1b — over-level total runs
    ax2 = axes[1]
    over_runs = (deliveries.groupby(["match_id", "over"])["total_runs"]
                 .sum().reset_index()["total_runs"])
    ax2.hist(over_runs, bins=30, color=IPL_BLUE, edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Runs Scored in an Over")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Runs per Over (Histogram)")
    fig.tight_layout()
    return _savefig("01_histogram", fig)


# ─────────────────────────────────────────────────────────────────
# 2. BAR GRAPH — Top-10 match-winning teams
# ─────────────────────────────────────────────────────────────────
def plot_bar(matches):
    fig, ax = plt.subplots(figsize=(12, 6))
    wins = matches["winner"].value_counts().head(10)
    bars = ax.barh(wins.index[::-1], wins.values[::-1],
                   color=sns.color_palette("Blues_r", len(wins)))
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Number of Wins")
    ax.set_title("Top 10 IPL Match-Winning Teams")
    fig.tight_layout()
    return _savefig("02_bar_team_wins", fig)


# ─────────────────────────────────────────────────────────────────
# 3. PIE CHART — Toss decision split
# ─────────────────────────────────────────────────────────────────
def plot_pie(matches):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Toss Analysis", fontsize=15, fontweight="bold")

    # 3a — bat vs field
    td = matches["toss_decision"].value_counts()
    axes[0].pie(td.values, labels=td.index, autopct="%1.1f%%",
                colors=[IPL_BLUE, IPL_GOLD], startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Toss Decision Split")

    # 3b — toss winner == match winner?
    toss_win = (matches["toss_winner"] == matches["winner"]).value_counts()
    axes[1].pie(toss_win.values,
                labels=["Toss winner\nLost match", "Toss winner\nWon match"],
                autopct="%1.1f%%",
                colors=["#E74C3C", "#2ECC71"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Toss Winner → Match Winner?")
    fig.tight_layout()
    return _savefig("03_pie_toss", fig)


# ─────────────────────────────────────────────────────────────────
# 4. SCATTER PLOT — First-innings score vs. Result + toss decision
# ─────────────────────────────────────────────────────────────────
def plot_scatter(matches, deliveries):
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby("match_id")["total_runs"].sum().reset_index()
            .rename(columns={"total_runs": "first_innings_score"}))
    merged = matches.merge(inn1, left_on="id", right_on="match_id", how="inner")
    merged["toss_win_match"] = (merged["toss_winner"] == merged["winner"]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Score vs Match Outcome", fontsize=15, fontweight="bold")

    # 4a — score vs. win margin (runs)
    ax = axes[0]
    margin_data = merged[merged["result"] == "runs"].copy()
    margin_data["win_by_runs"] = pd.to_numeric(margin_data["result_margin"], errors="coerce")
    sc = ax.scatter(
        margin_data["first_innings_score"], margin_data["win_by_runs"],
        c=margin_data["first_innings_score"], cmap="YlOrRd",
        alpha=0.6, s=40, edgecolors="none"
    )
    fig.colorbar(sc, ax=ax, label="1st Innings Score")
    ax.set_xlabel("1st Innings Score")
    ax.set_ylabel("Win Margin (Runs)")
    ax.set_title("Score vs Win by Runs")

    # 4b — score vs season (jittered)
    ax2 = axes[1]
    jitter = np.random.uniform(-0.3, 0.3, len(merged))
    ax2.scatter(merged["season"] + jitter, merged["first_innings_score"],
                c=merged["toss_win_match"], cmap="coolwarm",
                alpha=0.4, s=20, edgecolors="none")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("1st Innings Score")
    ax2.set_title("Scores by Season (colour = toss won match)")
    fig.tight_layout()
    return _savefig("04_scatter", fig)


# ─────────────────────────────────────────────────────────────────
# 5. BOX-WHISKER — First-innings score per team
# ─────────────────────────────────────────────────────────────────
def plot_boxwhisker(deliveries):
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby(["match_id", "batting_team"])["total_runs"]
            .sum().reset_index())

    team_counts = inn1["batting_team"].value_counts()
    top_teams   = team_counts[team_counts >= 20].index
    inn1 = inn1[inn1["batting_team"].isin(top_teams)]

    order = (inn1.groupby("batting_team")["total_runs"]
             .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=inn1, x="batting_team", y="total_runs",
                order=order, palette=PALETTE, ax=ax,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("1st Innings Runs")
    ax.set_title("Box-Whisker: 1st Innings Score by Team")
    ax.axhline(inn1["total_runs"].median(), ls="--", color="red",
               lw=1.2, label="Overall Median")
    ax.legend()
    fig.tight_layout()
    return _savefig("05_boxwhisker", fig)


# ─────────────────────────────────────────────────────────────────
# 6. KDE PLOT — First-innings score density per era
# ─────────────────────────────────────────────────────────────────
def plot_kde(matches, deliveries):
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby("match_id")["total_runs"].sum().reset_index()
            .rename(columns={"total_runs": "score"}))
    merged = inn1.merge(matches[["id", "season"]], left_on="match_id", right_on="id")

    def era(s):
        if s <= 2012: return "Early (2008-12)"
        if s <= 2017: return "Mid   (2013-17)"
        return "Modern(2018+)"

    merged["era"] = merged["season"].apply(era)

    fig, ax = plt.subplots(figsize=(11, 5))
    for era_name, grp in merged.groupby("era"):
        sns.kdeplot(grp["score"], ax=ax, label=era_name, fill=True, alpha=0.3, linewidth=2)
    ax.set_xlabel("1st Innings Score")
    ax.set_ylabel("Density")
    ax.set_title("KDE: 1st Innings Score Density by IPL Era")
    ax.legend(title="Era")
    fig.tight_layout()
    return _savefig("06_kde", fig)


# ─────────────────────────────────────────────────────────────────
# 7. HEATMAP — Head-to-head win matrix (ADVANCED)
# ─────────────────────────────────────────────────────────────────
def plot_heatmap(matches):
    teams = sorted(set(matches["team1"]) | set(matches["team2"]))
    matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=float)
    totals = pd.DataFrame(0, index=teams, columns=teams, dtype=float)

    for _, row in matches.iterrows():
        t1, t2, w = row["team1"], row["team2"], row.get("winner", np.nan)
        if pd.isna(w): continue
        totals.loc[t1, t2] += 1
        totals.loc[t2, t1] += 1
        if w == t1: matrix.loc[t1, t2] += 1
        elif w == t2: matrix.loc[t2, t1] += 1

    # Win-rate matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        wr_matrix = matrix.div(totals).fillna(np.nan)

    # Filter: keep teams with >= 10 matches total
    active = [t for t in teams if totals.loc[t].sum() >= 10]
    wr_matrix = wr_matrix.loc[active, active]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = wr_matrix.isnull() | (wr_matrix == 0)
    sns.heatmap(
        wr_matrix, annot=True, fmt=".0%", cmap="RdYlGn",
        linewidths=0.5, linecolor="white",
        mask=mask, ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Win Rate (row vs col)"},
        annot_kws={"size": 8}
    )
    ax.set_title("Head-to-Head Win Rate Matrix\n(row team wins % against col team)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    return _savefig("07_heatmap_h2h", fig)


# ─────────────────────────────────────────────────────────────────
# 8. VIOLIN PLOT — Season-wise 1st innings scores (ADVANCED)
# ─────────────────────────────────────────────────────────────────
def plot_violin(matches, deliveries):
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby("match_id")["total_runs"].sum().reset_index()
            .rename(columns={"total_runs": "score"}))
    merged = inn1.merge(matches[["id", "season"]], left_on="match_id", right_on="id")
    merged["season"] = merged["season"].astype(str)
    seasons = sorted(merged["season"].unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.violinplot(data=merged, x="season", y="score",
                   order=seasons, palette="husl", inner="quartile",
                   cut=0, linewidth=0.8, ax=ax)
    ax.axhline(merged["score"].mean(), ls="--", color="red",
               lw=1.5, label=f"Overall Mean ({merged['score'].mean():.0f})")
    ax.set_xlabel("Season")
    ax.set_ylabel("1st Innings Score")
    ax.set_title("Violin Plot: 1st Innings Score by Season")
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return _savefig("08_violin_season", fig)


# ─────────────────────────────────────────────────────────────────
# 9. PAIR PLOT — Winner-prediction features (ADVANCED)
# ─────────────────────────────────────────────────────────────────
def plot_pairplot(matches, deliveries):
    from sklearn.preprocessing import LabelEncoder

    inn1_map = (deliveries[deliveries["inning"] == 1]
                .groupby(["match_id", "batting_team"])["total_runs"]
                .sum().to_dict())

    rows = []
    for _, row in matches.iterrows():
        t1, t2, w = row["team1"], row["team2"], row.get("winner", np.nan)
        if pd.isna(w): continue
        target = 1 if w == t1 else 0
        rows.append({
            "t1_score": inn1_map.get((row["id"], t1), np.nan),
            "t2_score": inn1_map.get((row["id"], t2), np.nan),
            "toss_is_t1":  1 if row["toss_winner"] == t1 else 0,
            "toss_bat":    1 if row["toss_decision"] == "bat" else 0,
            "target":      target,
        })

    df = pd.DataFrame(rows).dropna()

    if df.empty:
        print("  ⚠️  Skipping pairplot — no valid data available")
        return None

    # Safe sampling
    df = df.sample(min(600, len(df)), random_state=42)

    df["winner"] = df["target"].map({1: "Team1", 0: "Team2"})

    g = sns.pairplot(
        df[["t1_score", "t2_score", "toss_is_t1", "toss_bat", "winner"]],
        hue="winner", palette={"Team1": IPL_BLUE, "Team2": "#E74C3C"},
        diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20},
        corner=True
    )
    g.figure.suptitle("Pair Plot: Winner Prediction Features", y=1.02, fontsize=14)
    g.figure.tight_layout()
    return _savefig("09_pairplot", g.figure)


# ─────────────────────────────────────────────────────────────────
# 10. STACKED BAR — Toss decision vs win rate
# ─────────────────────────────────────────────────────────────────
def plot_stacked_bar(matches):
    m = matches.dropna(subset=["winner", "toss_winner", "toss_decision"]).copy()
    m["toss_won_match"] = (m["toss_winner"] == m["winner"])

    agg = (m.groupby("toss_decision")["toss_won_match"]
           .agg(["sum", "count"]).reset_index())
    agg.columns = ["decision", "won", "total"]
    agg["lost"] = agg["total"] - agg["won"]
    agg["won_pct"]  = agg["won"]  / agg["total"] * 100
    agg["lost_pct"] = agg["lost"] / agg["total"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    bars_w = ax.bar(x, agg["won_pct"],  color="#2ECC71", label="Won Match")
    bars_l = ax.bar(x, agg["lost_pct"], bottom=agg["won_pct"], color="#E74C3C", label="Lost Match")
    ax.bar_label(bars_w, labels=[f"{v:.1f}%" for v in agg["won_pct"]], label_type="center", color="white", fontweight="bold")
    ax.bar_label(bars_l, labels=[f"{v:.1f}%" for v in agg["lost_pct"]], label_type="center", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["decision"].str.title(), fontsize=12)
    ax.set_ylabel("Percentage of Matches")
    ax.set_title("Stacked Bar: Toss Decision vs Match Win Rate")
    ax.legend()
    ax.set_ylim(0, 115)
    fig.tight_layout()
    return _savefig("10_stacked_bar_toss", fig)


# ─────────────────────────────────────────────────────────────────
# 11. BUBBLE CHART — Venue stats (ADVANCED)
# ─────────────────────────────────────────────────────────────────
def plot_bubble(matches, deliveries):
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby("match_id")["total_runs"].sum().reset_index()
            .rename(columns={"total_runs": "score"}))
    merged = matches.merge(inn1, left_on="id", right_on="match_id", how="inner")

    venue_stats = merged.groupby("venue").agg(
        matches  = ("id", "count"),
        avg_score= ("score", "mean"),
    ).reset_index()

    # Toss advantage per venue
    toss_adv = (
        merged.groupby("venue")
        .apply(lambda df: (df["toss_winner"] == df["winner"]).mean())
        .reset_index(name="toss_win_rate")
    )
    venue_stats = venue_stats.merge(toss_adv, on="venue")
    venue_stats = venue_stats[venue_stats["matches"] >= 10]
    venue_stats = venue_stats.sort_values("matches", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(13, 7))
    scatter = ax.scatter(
        venue_stats["avg_score"],
        venue_stats["toss_win_rate"] * 100,
        s=venue_stats["matches"] * 10,
        c=venue_stats["matches"],
        cmap="plasma", alpha=0.75, edgecolors="white", linewidths=0.8
    )
    for _, r in venue_stats.iterrows():
        ax.annotate(r["venue"], (r["avg_score"], r["toss_win_rate"] * 100),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Total Matches")
    ax.set_xlabel("Average 1st Innings Score")
    ax.set_ylabel("Toss Winner → Match Win Rate (%)")
    ax.set_title("Bubble Chart: Venue Analysis\n(bubble size = number of matches)")
    ax.axhline(50, ls="--", color="gray", lw=1)
    fig.tight_layout()
    return _savefig("11_bubble_venue", fig)


# ─────────────────────────────────────────────────────────────────
# 12. ROLLING FORM — Top-4 teams rolling 10-match win rate (ADVANCED)
# ─────────────────────────────────────────────────────────────────
def plot_rolling_form(matches):
    top4 = matches["winner"].value_counts().head(4).index.tolist()
    matches_sorted = matches.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = sns.color_palette("tab10", 4)

    for i, team in enumerate(top4):
        team_matches = matches_sorted[
            (matches_sorted["team1"] == team) | (matches_sorted["team2"] == team)
        ].copy()
        team_matches["won"] = (team_matches["winner"] == team).astype(int)
        team_matches["rolling_wr"] = (team_matches["won"]
                                       .rolling(10, min_periods=5)
                                       .mean() * 100)
        ax.plot(
            range(len(team_matches)),
            team_matches["rolling_wr"],
            label=team, color=colors[i], lw=2, alpha=0.85
        )

    ax.set_xlabel("Match Number (chronological)")
    ax.set_ylabel("10-Match Rolling Win Rate (%)")
    ax.set_title("Rolling Form: Top-4 IPL Teams (10-match window)")
    ax.axhline(50, ls="--", color="gray", lw=1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _savefig("12_rolling_form", fig)


# ─────────────────────────────────────────────────────────────────
# 13. CORRELATION MATRIX — Feature correlations
# ─────────────────────────────────────────────────────────────────
def plot_correlation(matches, deliveries):
    inn1_map = (deliveries[deliveries["inning"] == 1]
                .groupby(["match_id", "batting_team"])["total_runs"]
                .sum().to_dict())

    rows = []
    for _, row in matches.iterrows():
        t1, t2, w = row["team1"], row["team2"], row.get("winner", np.nan)
        if pd.isna(w): continue
        t1s = inn1_map.get((row["id"], t1), np.nan)
        t2s = inn1_map.get((row["id"], t2), np.nan)
        rows.append({
            "target":     1 if w == t1 else 0,
            "t1_score":   t1s,
            "t2_score":   t2s,
            "score_diff": (t1s or 0) - (t2s or 0),
            "toss_is_t1": 1 if row["toss_winner"] == t1 else 0,
            "toss_bat":   1 if row["toss_decision"] == "bat" else 0,
        })

    df = pd.DataFrame(rows).dropna()
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix: Match Features vs Winner")
    fig.tight_layout()
    return _savefig("13_correlation_matrix", fig)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def perform_eda():
    print("\n📊 IPL EDA — Generating all visualizations...")
    print("─" * 55)

    matches, deliveries = load_data()

    paths = []
    paths.append(plot_histogram(deliveries))
    paths.append(plot_bar(matches))
    paths.append(plot_pie(matches))
    paths.append(plot_scatter(matches, deliveries))
    paths.append(plot_boxwhisker(deliveries))
    paths.append(plot_kde(matches, deliveries))
    paths.append(plot_heatmap(matches))
    paths.append(plot_violin(matches, deliveries))
    # paths.append(plot_pairplot(matches, deliveries))
    paths.append(plot_stacked_bar(matches))
    paths.append(plot_bubble(matches, deliveries))
    paths.append(plot_rolling_form(matches))
    paths.append(plot_correlation(matches, deliveries))

    print(f"\n✅  {len(paths)} plots saved to  {OUT}/")
    print("─" * 55)
    return paths


if __name__ == "__main__":
    perform_eda()