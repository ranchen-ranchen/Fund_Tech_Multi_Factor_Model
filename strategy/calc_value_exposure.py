# -*- coding: utf-8 -*-
"""
Compute the value factor exposure (standardized inverse of PB) for individual stocks
on a specified cross-section date, and visualize the distribution of factor exposure
(number of stocks) using a seaborn histogram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def value_factor_exposure(
    df: pd.DataFrame,
    target_date=None,
    *,
    code_col: str = "code",
    pb_col: str = "pbMRQ",
    date_col: str = "date",
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    clip_std: float = 3.0,
    n_bins: int = 30,
    plot: bool = True,
    ax=None,
):
    """
    Compute the value factor exposure (1/PB) for stocks on a specified cross-section,
    standardize it cross-sectionally, and visualize the factor exposure distribution
    using a seaborn histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least `code_col` and `pb_col`. If it also contains
        `date_col`, the cross-section is filtered by `target_date`.
    target_date : str / pd.Timestamp / None
        The specified cross-section date. Used to filter rows if `date_col` exists;
        otherwise used only for the chart annotation.
    code_col, pb_col, date_col : str
        Column names, default "code" / "pbMRQ" / "date".
    winsorize : bool
        Whether to winsorize the raw 1/PB values by quantiles (default True).
    lower_q, upper_q : float
        Quantiles used for winsorization (default 1% / 99%).
    clip_std : float
        After standardization, clip values to +/- `clip_std` to limit extreme values
        in the plot (default 3).
    n_bins : int
        Number of histogram bins.
    plot : bool
        Whether to draw the plot.
    ax : matplotlib.axes.Axes or None
        An existing axes to draw on; if None, a new one is created.

    Returns
    -------
    pd.DataFrame
        Cross-section result with columns: code, pbMRQ, value_raw, value_z,
        sorted by value_z in descending order.
    """
    data = df.copy()

    # ---------- 1. Select the specified cross-section ----------
    if date_col in data.columns:
        if target_date is None:
            raise ValueError(
                f"Data contains '{date_col}' column; please specify a cross-section date via target_date"
            )
        target_ts = pd.Timestamp(target_date)
        data = data.loc[pd.to_datetime(data[date_col]) == target_ts].copy()
        if data.empty:
            raise ValueError(f"No data found on {target_ts.date()}")

    # ---------- 2. Compute raw value exposure 1/PB ----------
    data[pb_col] = pd.to_numeric(data[pb_col], errors="coerce")
    valid = data[pb_col].notna() & (data[pb_col] > 0)
    data = data.loc[valid].copy()                      # drop stocks with missing or non-positive PB
    data["value_raw"] = 1.0 / data[pb_col]             # value factor: inverse of price-to-book ratio

    n = len(data)
    if n < 2:
        raise ValueError(
            f"Insufficient valid stocks (currently {n}); cannot perform cross-sectional standardization"
        )

    # ---------- 3. Winsorize + standardize (Z-Score) ----------
    raw = data["value_raw"]
    if winsorize:
        lo, hi = raw.quantile(lower_q), raw.quantile(upper_q)
        raw = raw.clip(lo, hi)

    mu = raw.mean()
    sigma = raw.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError(
            "Cross-sectional standard deviation of factor exposure is 0; cannot standardize"
        )

    z = (raw - mu) / sigma
    z = z.clip(-clip_std, clip_std)
    z = (z - z.mean()) / z.std(ddof=0)                 # standardize again after clipping

    data["value_z"] = z
    data = data.sort_values("value_z", ascending=False).reset_index(drop=True)

    # ---------- 4. Seaborn histogram visualization ----------
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        z_clean = data["value_z"].dropna()

        sns.histplot(
            z_clean,
            bins=n_bins,
            stat="count",
            color="steelblue",
            edgecolor="white",
            ax=ax,
        )

        # Overlay a standard normal reference density (scaled to counts)
        if len(z_clean) > 1:
            xs = np.linspace(z_clean.min(), z_clean.max(), 300)
            bin_width = (z_clean.max() - z_clean.min()) / n_bins
            pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)
            ax.plot(
                xs,
                len(z_clean) * bin_width * pdf,
                color="crimson",
                lw=1.8,
                label="Standard normal density (scaled to counts)",
            )

        ax.axvline(0.0, color="gray", ls="--", lw=1, label="Mean = 0")
        ax.set_xlabel("Value Factor Exposure (Standardized 1/PB)")
        ax.set_ylabel("Number of Stocks")
        date_str = f" @ {pd.Timestamp(target_date).date()}" if target_date is not None else ""
        ax.set_title(f"Value Factor Exposure Distribution (N = {len(z_clean)}){date_str}")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

    return data


# ============================ Example / self-test ============================
if __name__ == "__main__":
    # Font settings
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    # --- Construct simulated data: two cross-section dates x 300 stocks ---
    rng = np.random.default_rng(42)
    dates = pd.to_datetime(["2024-06-28", "2024-07-31"])
    codes = [f"{i:06d}" for i in range(1, 301)]

    frames = []
    for d in dates:
        pb = np.exp(rng.normal(0.0, 0.8, size=len(codes))) * 3.0   # log-normal PB values
        frames.append(pd.DataFrame({"date": d, "code": codes, "pbMRQ": pb}))
    df = pd.concat(frames, ignore_index=True)

    # Artificially create some invalid PB values (missing / negative)
    df.loc[df.sample(8, random_state=1).index, "pbMRQ"] = np.nan
    df.loc[df.sample(6, random_state=2).index, "pbMRQ"] = -1.0

    # --- Compute value factor exposure for the specified cross-section ---
    result = value_factor_exposure(df, target_date="2024-06-28")

    print(result.head(10))
    print(f"\nNumber of valid stocks: {len(result)}")
    print(f"Exposure mean = {result['value_z'].mean():.6f}, "
          f"std = {result['value_z'].std(ddof=0):.6f}")

    plt.show()



