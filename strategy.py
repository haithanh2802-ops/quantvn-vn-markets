import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantvn import client
from quantvn.vn.data import get_derivatives_hist
from quantvn.vn.metrics import Metrics, Backtest_Derivates


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    # Chuan hoa thoi gian
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Tinh VWAP intraday
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = df["typical_price"] * df["volume"]
    df["cum_pv"] = df.groupby("Date")["pv"].cumsum()
    df["cum_volume"] = df.groupby("Date")["volume"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_volume"]

    # Expected cumulative volume curve: trung binh 20 ngay truoc tai cung phut
    df["minute_no"] = df.groupby("Date").cumcount() + 1

    daily_curve = df[["Date", "minute_no", "cum_volume"]].copy()
    expected_list = []
    all_dates = sorted(df["Date"].unique())

    for current_date in all_dates:
        past_dates = [d for d in all_dates if d < current_date][-20:]

        if len(past_dates) < 20:
            continue

        expected = (
            daily_curve[daily_curve["Date"].isin(past_dates)]
            .groupby("minute_no")["cum_volume"]
            .mean()
            .reset_index()
            .rename(columns={"cum_volume": "expected_cum_volume"})
        )

        expected["Date"] = current_date
        expected_list.append(expected)

    if expected_list:
        expected_df = pd.concat(expected_list, ignore_index=True)
        df = df.merge(expected_df, on=["Date", "minute_no"], how="left")
    else:
        df["expected_cum_volume"] = np.nan

    # Features
    df["relative_volume"] = df["cum_volume"] / df["expected_cum_volume"]
    df["vwap_gap"] = (df["Close"] - df["vwap"]) / df["vwap"]
    df["momentum"] = df.groupby("Date")["Close"].pct_change(15)

    # Tin hieu giao dich
    df["position"] = 0

    long_condition = (
        (df["relative_volume"] > 1.2)
        & (df["vwap_gap"] > 0)
        & (df["momentum"] > 0)
    )

    short_condition = (
        (df["relative_volume"] > 1.2)
        & (df["vwap_gap"] < 0)
        & (df["momentum"] < 0)
    )

    df.loc[long_condition, "position"] = 1
    df.loc[short_condition, "position"] = -1

    # Khong giu vi the sau 14:30
    df["bar_time"] = df["Datetime"].dt.time
    exit_time = pd.to_datetime("14:30:00").time()
    df.loc[df["bar_time"] >= exit_time, "position"] = 0
    return df


if __name__ == "__main__":
    client(apikey="I2IDzLhhPBWkZ2Mab0yiVp1ZVZnAXYh3JA3T9hofFKo6Fe60MamlE0J0WscyAjClYI4YWmoiRhCu9dM9AwWZQQFbmLZxGYmyK4d8L5zziGjDoN8gWNRzzUaoilyS2jXY")

    # Lay du lieu
    df = get_derivatives_hist("VN30F1M", "1m")
    df = gen_position(df)

    # Backtest
    backtest = Backtest_Derivates(df, pnl_type="after_fees")
    metrics = Metrics(backtest)

    # Ket qua
    print("=" * 60)
    print("VWAP RELATIVE VOLUME MOMENTUM - VN30F1M (1min)")
    print("=" * 60)
    print(f"Final PnL: {backtest.PNL().iloc[-1]:,.0f} VND")
    print(f"Sharpe Ratio: {metrics.sharpe():.3f}")
    print(f"Win Rate: {metrics.win_rate() * 100:.2f}%")
    print(f"Max Drawdown: {metrics.max_drawdown() * 100:.2f}%")
    print(f"Profit Factor: {metrics.profit_factor():.3f}")

    # Ve bieu do
    backtest.plot_PNL(title="VN30F1M - VWAP Relative Volume Momentum")
