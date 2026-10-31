import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_volatility_exposure(
    stock_data,
    start_date,
    end_date,
    stock_names=None,
    annualize=True,
    min_obs=20,
    standardize=True,
    plot=True,
    bins=20,
    figsize=(10, 6),
):
    """
    Compute the volatility factor exposure of a set of stocks over a
    specified period, standardize it cross-sectionally, and plot a
    histogram of the distribution.

    Parameters
    ----------
    stock_data : list[pd.DataFrame] or dict[str, pd.DataFrame]
        Each DataFrame must contain 'date' and 'close' columns.
        If a list, stock_names is used to name the stocks; if a dict,
        its keys are used automatically as stock names.
    start_date : str or pd.Timestamp
        Start date of the period (inclusive).
    end_date : str or pd.Timestamp
        End date of the period (inclusive).
    stock_names : list[str], optional
        Stock names corresponding one-to-one to stock_data (list form).
    annualize : bool
        Whether to annualize daily volatility (multiply by sqrt(252)).
    min_obs : int
        Minimum number of valid return observations in the period;
        otherwise the raw exposure is set to NaN.
    standardize : bool
        Whether to apply cross-sectional z-score standardization.
    plot : bool
        Whether to draw the histogram.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.

    Returns
    -------
    exposure_df : pd.DataFrame
        Columns: ['stock', 'vol_raw', 'vol_exposure']
        - vol_raw: raw volatility exposure (std of daily returns,
          possibly annualized)
        - vol_exposure: standardized factor exposure
    fig : matplotlib.figure.Figure or None
        The figure object if plot=True, otherwise None.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Normalize to [(name, DataFrame), ...]
    if isinstance(stock_data, dict):
        items = list(stock_data.items())
    else:
        if stock_names is None:
            stock_names = [f"stock_{i}" for i in range(len(stock_data))]
        if len(stock_names) != len(stock_data):
            raise ValueError("Length of stock_names does not match stock_data")
        items = list(zip(stock_names, stock_data))

    records = []
    for name, df in items:
        if not {"date", "close"}.issubset(df.columns):
            raise ValueError(
                f"Data for stock {name} is missing 'date' or 'close' column"
            )

        tmp = df[["date", "close"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        tmp = tmp[(tmp["date"] >= start) & (tmp["date"] <= end)]

        vol_raw = np.nan
        if len(tmp) >= min_obs + 1:
            ret = tmp["close"].pct_change().dropna()
            if len(ret) >= min_obs:
                vol_raw = ret.std(ddof=1)  # standard deviation of daily returns
                if annualize:
                    vol_raw *= np.sqrt(252)

        records.append({"stock": name, "vol_raw": vol_raw})

    exposure_df = pd.DataFrame(records)

    # ---- Cross-sectional standardization (z-score) ----
    if standardize:
        valid = exposure_df["vol_raw"].dropna()
        if len(valid) > 1 and valid.std(ddof=1) > 0:
            mu, sigma = valid.mean(), valid.std(ddof=1)
            exposure_df["vol_exposure"] = (exposure_df["vol_raw"] - mu) / sigma
        else:
            exposure_df["vol_exposure"] = np.nan
    else:
        exposure_df["vol_exposure"] = exposure_df["vol_raw"]

    # ---- Histogram visualization ----
    fig = None
    if plot:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        data = exposure_df["vol_exposure"].dropna()
        sns.histplot(
            x=data,
            bins=bins,
            kde=True,
            color="steelblue",
            edgecolor="white",
            ax=ax,
        )

        ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2,
                   label="Mean (0)")
        ax.set_xlabel("Volatility Factor Exposure (Standardized)")
        ax.set_ylabel("Number of Stocks")
        ax.set_title(
            f"Cross-Sectional Distribution of Volatility Factor Exposure "
            f"({start.date()} to {end.date()}, "
            f"{len(data)} valid stocks)"
        )
        ax.legend()
        plt.tight_layout()

    return exposure_df, fig


# ======================================================================
# Example: generate simulated data and call the function
# ======================================================================
if __name__ == "__main__":
    np.random.seed(42)

    dates = pd.bdate_range("2023-01-01", "2023-12-31")
    n_stocks = 80

    stock_frames = []
    names = []
    for i in range(n_stocks):
        # Different volatility levels for each stock
        sigma = np.random.uniform(0.008, 0.035)
        rets = np.random.normal(0.0003, sigma, size=len(dates))
        price = 100 * np.cumprod(1 + rets)

        stock_frames.append(
            pd.DataFrame({"date": dates, "close": price})
        )
        names.append(f"STK{i:03d}")

    exposure_df, fig = compute_volatility_exposure(
        stock_data=stock_frames,
        start_date="2023-02-01",
        end_date="2023-09-30",
        stock_names=names,
        annualize=True,
        standardize=True,
        plot=True,
        bins=20,
    )

    print("Summary statistics of standardized exposure:")
    print(exposure_df["vol_exposure"].describe())
    print("\nFirst 10 rows:")
    print(exposure_df.head(10).to_string(index=False))

    plt.show()
