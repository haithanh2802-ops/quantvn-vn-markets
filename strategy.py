from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Strategy parameters.
MIN_HISTORY_DAYS = 20
RVOL_THRESHOLD = 1.10
VWAP_BAND = 0.5
VWAP_SLOPE_BARS = 3
VWAP_SLOPE_MIN = 0.05
EXECUTION_LAG_BARS = 1
TRADE_START = "09:15:00"
FLAT_AFTER = "14:25:00"
STOP_LOSS_POINTS = 3.0
TIME_STOP_BARS = 5
COOLDOWN_BARS = 3

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    # Logic chiến lược:
    # Long VN30F1M khi giá nằm trên VWAP đủ xa và volume tích lũy đang chạy
    # nhanh hơn đường cong volume bình thường trong ngày.
    #
    # Short VN30F1M khi giá nằm dưới VWAP đủ xa và volume tích lũy cũng đang
    # chạy nhanh hơn bình thường.
    #
    # position được dịch sang bar kế tiếp để tránh dùng tín hiệu của chính bar
    # hiện tại để khớp lệnh trong cùng bar.
    df = _prepare_data(df)
    df = _add_vwap(df)
    df = _add_cum_rvol(df)

    df["vwap_dev"] = df["Close"] - df["vwap"]
    df["vwap_dev_pct"] = df["vwap_dev"] / df["vwap"]
    df["vwap_slope"] = df.groupby("session")["vwap"].diff(VWAP_SLOPE_BARS)

    long_signal = (
        (df["vwap_dev"] > VWAP_BAND)
        & (df["cum_rvol"] > RVOL_THRESHOLD)
        & (df["vwap_slope"] > VWAP_SLOPE_MIN)
    )
    short_signal = (
        (df["vwap_dev"] < -VWAP_BAND)
        & (df["cum_rvol"] > RVOL_THRESHOLD)
        & (df["vwap_slope"] < -VWAP_SLOPE_MIN)
    )

    df["signal"] = np.select([long_signal, short_signal], [1, -1], default=0)

    # Không giao dịch khi chưa có đủ lịch sử để ước lượng volume curve.
    df.loc[df["history_days"] < MIN_HISTORY_DAYS, "signal"] = 0

    # Không mở vị thế sau mốc này để tránh nhiễu cuối phiên.
    trade_start = pd.to_datetime(TRADE_START).time()
    flat_after = pd.to_datetime(FLAT_AFTER).time()
    df.loc[df["bar_time"] < trade_start, "signal"] = 0
    df.loc[df["bar_time"] > flat_after, "signal"] = 0

    df["target_position"] = (
        df.groupby("session")["signal"]
        .shift(EXECUTION_LAG_BARS)
        .fillna(0)
        .astype(int)
    )
    df["position"] = _build_position_with_risk_controls(df)
    df.loc[df["bar_time"] > flat_after, "position"] = 0

    return df.drop(columns=["session", "bar_time"])


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hóa dữ liệu đầu vào về đúng format intraday OHLCV.
    required_cols = {"Date", "time", "Open", "High", "Low", "Close", "volume"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    out = df.copy()
    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    else:
        out["Datetime"] = pd.to_datetime(
            out["Date"].astype(str) + " " + out["time"].astype(str),
            errors="coerce",
        )

    out = out.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close", "volume"])

    out["Date"] = out["Datetime"].dt.date.astype(str)
    out["time"] = out["Datetime"].dt.strftime("%H:%M:%S")
    out["session"] = out["Datetime"].dt.date
    out["bar_time"] = out["Datetime"].dt.time
    return out


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    # VWAP trong ngày = tổng typical_price * volume / tổng volume.
    out = df.copy()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["dollar_volume"] = typical_price * out["volume"]

    group = out.groupby("session", sort=False)
    out["cum_volume"] = group["volume"].cumsum()
    out["cum_dollar_volume"] = group["dollar_volume"].cumsum()
    out["vwap"] = out["cum_dollar_volume"] / out["cum_volume"].replace(0, np.nan)
    return out


def _add_cum_rvol(df: pd.DataFrame) -> pd.DataFrame:
    # Tạo expected cumulative volume theo từng mốc thời gian từ các phiên trước.
    # Dùng median expanding và shift(1) để không nhìn vào dữ liệu tương lai.
    out = df.copy()
    volume_curve = (
        out.pivot_table(
            index="session",
            columns="time",
            values="cum_volume",
            aggfunc="last",
        )
        .sort_index()
        .ffill(axis=1)
    )

    expected_volume = (
        volume_curve.expanding(min_periods=MIN_HISTORY_DAYS).median().shift(1)
    )
    history_days = (
        volume_curve.notna().expanding(min_periods=1).count().shift(1).fillna(0)
    )

    features = pd.concat(
        [
            expected_volume.stack(future_stack=True).rename("expected_cum_volume"),
            history_days.stack(future_stack=True).rename("history_days"),
        ],
        axis=1,
    ).reset_index()

    out = out.merge(features, on=["session", "time"], how="left")
    out["cum_rvol"] = out["cum_volume"] / out["expected_cum_volume"].replace(0, np.nan)
    out["cum_rvol"] = out["cum_rvol"].replace([np.inf, -np.inf], np.nan)
    out["history_days"] = out["history_days"].fillna(0)
    return out


def _build_position_with_risk_controls(df: pd.DataFrame) -> pd.Series:
    # Chuyển target_position thành position thật với stop-loss, time-stop,
    # cooldown sau lệnh thua, và đóng vị thế khi signal không còn hợp lệ.
    positions = pd.Series(0, index=df.index, dtype=int)

    for _, session_df in df.groupby("session", sort=False):
        pos = 0
        entry_price = np.nan
        bars_held = 0
        cooldown = 0

        for idx, row in session_df.iterrows():
            target = int(row["target_position"])
            close = float(row["Close"])

            if cooldown > 0:
                cooldown -= 1

            if pos != 0:
                bars_held += 1
                trade_pnl = (close - entry_price) * pos
                stop_hit = trade_pnl <= -STOP_LOSS_POINTS
                time_stop = bars_held >= TIME_STOP_BARS and trade_pnl <= 0
                signal_exit = target == 0 or target == -pos

                if stop_hit or time_stop or signal_exit:
                    if stop_hit or time_stop:
                        cooldown = COOLDOWN_BARS
                    pos = 0
                    entry_price = np.nan
                    bars_held = 0

            if pos == 0 and cooldown == 0 and target != 0:
                pos = target
                entry_price = close
                bars_held = 0

            positions.loc[idx] = pos
    return positions

def _sharpe_ratio(pnl: pd.Series, periods_per_year: int = 252) -> float:
    # Sharpe dùng thay đổi PnL theo ngày như chuỗi return đơn giản.
    pnl_change = pnl.diff().dropna()
    if pnl_change.empty:
        return 0.0

    std = pnl_change.std(ddof=1)
    if std == 0 or pd.isna(std):
        return 0.0

    return float((pnl_change.mean() / std) * np.sqrt(periods_per_year))


if __name__ == "__main__":
    # Phần này chỉ dùng để chạy nhanh backtest từ terminal.
    from quantvn import client
    from quantvn.vn.data.derivatives import get_hist
    from quantvn.vn.data.utils import APIKeyNotSetError
    from quantvn.vn.metrics import Backtest_Derivates

    api_key = os.getenv("QUANTVN_API_KEY")
    if api_key:
        client(apikey=api_key)

    try:
        data = get_hist("VN30F1M", "5m")
        result = gen_position(data)
        backtest = Backtest_Derivates(result, pnl_type="after_fees")
    except APIKeyNotSetError:
        print("API key is not set.")
        print('$env:QUANTVN_API_KEY="your_api_key_here"')
        print("& C:/Users/thanh/anaconda3/python.exe strategy.py")
    else:
        pnl = backtest.PNL()
        daily_pnl = backtest.daily_PNL()
        print(f"total_pnl: {pnl.iloc[-1]:.4f}")
        print(f"daily_pnl_last: {daily_pnl.iloc[-1]:.4f}")
        print(f"max_drawdown: {(pnl - pnl.cummax()).min():.4f}")
        print(f"daily_sharpe: {_sharpe_ratio(daily_pnl):.4f}")
        print(f"trade_count: {backtest.df['position'].diff().abs().fillna(0).sum():.0f}")
        print(f"active_bars: {(backtest.df['position'] != 0).sum():.0f}")
        print(f"avg_position_change_per_day: {backtest.avg_pos():.4f}")
        print(f"minimum_capital: {backtest.estimate_minimum_capital():.4f}")
