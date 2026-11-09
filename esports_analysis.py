import numpy as np
import pandas as pd


players = pd.DataFrame(
    [
        {"name": "Aidar", "team": "Falcons", "role": "Carry", "games": 20, "wins": 16, "score": 95},
        {"name": "Murat", "team": "Falcons", "role": "Support", "games": 18, "wins": 14, "score": 88},
        {"name": "Samat", "team": "Falcons", "role": "Mid", "games": 20, "wins": 15, "score": 92},
        {"name": "Ilyas", "team": "Falcons", "role": "Offlane", "games": 16, "wins": 12, "score": 84},
        {"name": "Timur", "team": "Wolves", "role": "Carry", "games": 20, "wins": 13, "score": 82},
        {"name": "Ruslan", "team": "Wolves", "role": "Support", "games": 18, "wins": 11, "score": 80},
        {"name": "Azamat", "team": "Wolves", "role": "Mid", "games": 16, "wins": 10, "score": 79},
        {"name": "Nurlan", "team": "Wolves", "role": "Offlane", "games": 20, "wins": 12, "score": 85},
        {"name": "Daniyar", "team": "Titans", "role": "Carry", "games": 20, "wins": 12, "score": 86},
        {"name": "Arman", "team": "Titans", "role": "Support", "games": 18, "wins": 10, "score": 78},
        {"name": "Erlan", "team": "Titans", "role": "Mid", "games": 16, "wins": 9, "score": 75},
        {"name": "Beka", "team": "Titans", "role": "Offlane", "games": 20, "wins": 11, "score": 83},
    ]
)

players["win_rate_pct"] = players["wins"] / players["games"] * 100

print("Процент побед каждого игрока:")
print(players[["name", "team", "win_rate_pct"]].round({"win_rate_pct": 2}).to_string(index=False))

print("\nИгроки с процентом побед больше 60%:")
print(
    players.loc[players["win_rate_pct"] > 60, ["name", "team", "win_rate_pct"]]
    .round({"win_rate_pct": 2})
    .to_string(index=False)
)

print("\nИгроки по убыванию score:")
print(players.sort_values("score", ascending=False).to_string(index=False))

best_player = players.loc[players["score"].idxmax()]
print(
    "\nЛучший игрок по score: "
    f"{best_player['name']} ({best_player['team']}), score = {best_player['score']}"
)
print(f"Средний score всех игроков: {players['score'].mean():.2f}")

team_report = players.groupby("team").agg(
    player_count=("name", "count"),
    avg_win_rate_pct=("win_rate_pct", "mean"),
    avg_score=("score", "mean"),
    max_score=("score", "max"),
    total_wins=("wins", "sum"),
    total_games=("games", "sum"),
)

print("\nКоличество игроков, средний и максимальный score по командам:")
print(
    team_report[
        ["player_count", "avg_score", "max_score"]
    ].round({"avg_score": 2}).to_string()
)

score_values = players["score"].to_numpy(dtype=float)
score_range = np.ptp(score_values)
normalized_score = np.divide(
    score_values - score_values.min(),
    score_range,
    out=np.zeros_like(score_values),
    where=score_range != 0,
) * 100
players["rating"] = 0.6 * players["win_rate_pct"].to_numpy() + 0.4 * normalized_score

print("\nИтоговый рейтинг игроков (60% процент побед + 40% нормированный score):")
print(
    players.sort_values("rating", ascending=False)[
        ["name", "team", "win_rate_pct", "score", "rating"]
    ]
    .round({"win_rate_pct": 2, "rating": 2})
    .to_string(index=False)
)

print("\nИтоговый аналитический отчёт по командам (groupby + agg):")
print(team_report.round({"avg_win_rate_pct": 2, "avg_score": 2}).to_string())

strongest_team = team_report.sort_values(
    ["avg_win_rate_pct", "avg_score"], ascending=False
).iloc[0]
strongest_team_name = team_report.sort_values(
    ["avg_win_rate_pct", "avg_score"], ascending=False
).index[0]
print(
    "\nСамая сильная команда по данным таблицы: "
    f"{strongest_team_name}. У неё самый высокий средний процент побед "
    f"({strongest_team['avg_win_rate_pct']:.2f}%) и средний score "
    f"{strongest_team['avg_score']:.2f}."
)
