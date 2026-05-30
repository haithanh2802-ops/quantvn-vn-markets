import os
import numpy as np
import pandas as pd

from quantvn import client
from quantvn.vn.data import get_derivatives_hist
from quantvn.vn.metrics import Backtest_Derivates, Metrics


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    # Strategy config. Keep these values here so the script is easy to tune.
    min_history_days = 20
    rvol_threshold = 1.10
    vwap_band = 0.5
    vwap_slope_bars = 3
    vwap_slope_min = 0.05
    execution_lag_bars = 1
    flat_after = pd.to_datetime("14:25:00").time()
    stop_loss_points = 3.0
    time_stop_bars = 5
    cooldown_bars = 3
    # Normalize datetime, sort bars, and keep one clean row order.
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    # Convert price and volume columns to numeric before calculating indicators.
    for col in ["High", "Low", "Close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["High", "Low", "Close", "volume"]).reset_index(drop=True)

    # Session fields used for intraday grouping.
    df["Date"] = df["Datetime"].dt.date
    df["time"] = df["Datetime"].dt.strftime("%H:%M:%S")
    df["bar_time"] = df["Datetime"].dt.time

    # Intraday VWAP: cumulative typical price * volume divided by cumulative volume.
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = typical_price * df["volume"]
    group = df.groupby("Date", sort=False)
    df["cum_volume"] = group["volume"].cumsum()
    df["cum_pv"] = group["pv"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_volume"].replace(0, np.nan)

    # Expected cumulative volume curve from prior sessions only.
    # This avoids looking into the future when calculating relative volume.
    volume_curve = (
        df.pivot_table(index="Date", columns="time", values="cum_volume", aggfunc="last")
        .sort_index()
        .ffill(axis=1)
    )
    expected_volume = volume_curve.expanding(min_periods=min_history_days).median().shift(1)
    history_days = volume_curve.notna().expanding(min_periods=1).count().shift(1).fillna(0)

    volume_features = pd.concat(
        [
            expected_volume.stack().rename("expected_cum_volume"),
            history_days.stack().rename("history_days"),
        ],
        axis=1,
    ).reset_index()

    df = df.merge(volume_features, on=["Date", "time"], how="left")
    df["cum_rvol"] = df["cum_volume"] / df["expected_cum_volume"].replace(0, np.nan)
    df["cum_rvol"] = df["cum_rvol"].replace([np.inf, -np.inf], np.nan)
    df["history_days"] = df["history_days"].fillna(0)

    # Price distance from VWAP and short VWAP slope are the directional filters.
    df["vwap_dev"] = df["Close"] - df["vwap"]
    df["vwap_slope"] = df.groupby("Date")["vwap"].diff(vwap_slope_bars)

    # Entry signals: trade only when volume is active and price is far enough from VWAP.
    long_signal = (
        (df["history_days"] >= min_history_days)
        & (df["bar_time"] <= flat_after)
        & (df["cum_rvol"] > rvol_threshold)
        & (df["vwap_dev"] > vwap_band)
        & (df["vwap_slope"] > vwap_slope_min)
    )
    short_signal = (
        (df["history_days"] >= min_history_days)
        & (df["bar_time"] <= flat_after)
        & (df["cum_rvol"] > rvol_threshold)
        & (df["vwap_dev"] < -vwap_band)
        & (df["vwap_slope"] < -vwap_slope_min)
    )

    # Delay the signal by one bar to avoid entering on information from the same bar.
    df["signal"] = np.select([long_signal, short_signal], [1, -1], default=0)
    df["target_position"] = (
        df.groupby("Date")["signal"].shift(execution_lag_bars).fillna(0).astype(int)
    )
    df["position"] = 0

    # Convert target positions into actual positions with stop loss, time stop,
    # cooldown after bad exits, and end-of-day flattening.
    for _, day_df in df.groupby("Date", sort=False):
        position = 0
        entry_price = np.nan
        bars_held = 0
        cooldown = 0

        for idx, row in day_df.iterrows():
            target = int(row["target_position"])
            close = float(row["Close"])

            if cooldown > 0:
                cooldown -= 1

            if position != 0:
                bars_held += 1
                trade_pnl = (close - entry_price) * position
                should_exit = (
                    (row["bar_time"] > flat_after)
                    or (trade_pnl <= -stop_loss_points)
                    or (bars_held >= time_stop_bars and trade_pnl <= 0)
                    or (target == 0)
                    or (target == -position)
                )

                if should_exit:
                    if trade_pnl <= 0:
                        cooldown = cooldown_bars
                    position = 0
                    entry_price = np.nan
                    bars_held = 0

            if position == 0 and cooldown == 0 and target != 0 and row["bar_time"] <= flat_after:
                position = target
                entry_price = close
                bars_held = 0

            df.at[idx, "position"] = position

    return df


if __name__ == "__main__":
    api_key = os.getenv("QUANTVN_API_KEY")
    client(api_key)
#getdata
    df = get_derivatives_hist("VN30F1M", "1m")
    df = gen_position(df)
#backtest
    backtest = Backtest_Derivates(df, pnl_type="after_fees")
    metrics = Metrics(backtest)
#result
    print("=" * 60)
    print("VWAP RVOL SLOPE - VN30F1M (1min)")
    print("=" * 60)
    print(f"Final PnL: {backtest.PNL().iloc[-1]:,.0f} VND")
    print(f"Sharpe Ratio: {metrics.sharpe():.3f}")
    print(f"Win Rate: {metrics.win_rate() * 100:.2f}%")
    print(f"Max Drawdown: {metrics.max_drawdown() * 100:.2f}%")
    print(f"Profit Factor: {metrics.profit_factor():.3f}")

    backtest.plot_PNL(title="VN30F1M - VWAP RVOL SLOPE")
