import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

load_dotenv()

def fetch_entries():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("""
        SELECT "moodScore", "sleepHours", "createdAt", "tags"
        FROM "Entry"
        WHERE "sleepHours" IS NOT NULL
        ORDER BY "createdAt" ASC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame(rows, columns=["mood", "sleep", "created_at", "tags"])
    return df

def build_features(df):
    df = df.copy()
    df["day_of_week"] = df["created_at"].dt.dayofweek  # 0=Monday
    df["has_work_tag"] = df["tags"].apply(lambda t: 1 if t and "work" in t else 0)
    df["has_health_tag"] = df["tags"].apply(lambda t: 1 if t and "health" in t else 0)
    df["has_social_tag"] = df["tags"].apply(lambda t: 1 if t and "social" in t else 0)

    # Rolling average of the last 3 entries' mood (a simple sense of recent trend)
    df["recent_mood_avg"] = df["mood"].rolling(window=3, min_periods=1).mean().shift(1)
    df["recent_mood_avg"] = df["recent_mood_avg"].fillna(df["mood"].mean())

    return df

def main():
    df = fetch_entries()
    print(f"Fetched {len(df)} entries")

    df = build_features(df)

    feature_cols = ["sleep", "day_of_week", "has_work_tag", "has_health_tag", "has_social_tag", "recent_mood_avg"]
    X = df[feature_cols]
    y = df["mood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)
    linear_mae = mean_absolute_error(y_test, linear_preds)

    # Tree-based model
    tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree_model.fit(X_train, y_train)
    tree_preds = tree_model.predict(X_test)
    tree_mae = mean_absolute_error(y_test, tree_preds)

    print(f"\nLinear Regression MAE: {linear_mae:.2f}")
    print(f"Decision Tree MAE: {tree_mae:.2f}")
    print("\n(MAE = average number of mood points the model's predictions are off by)")

    # Pick whichever model actually performed better on unseen data
    if linear_mae <= tree_mae:
        best_model = LinearRegression()
        best_name = "Linear Regression"
    else:
        best_model = DecisionTreeRegressor(max_depth=4, random_state=42)
        best_name = "Decision Tree"

    best_model.fit(X, y)  # retrain the winner on ALL data for the final saved version
    joblib.dump(best_model, "mood_model.joblib")
    print(f"\nSaved best model ({best_name}) to mood_model.joblib")

if __name__ == "__main__":
    main()