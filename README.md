# Quant Research Platform — AI Compute Chain Strategy, Backtesting & Factor Attribution

A research-oriented quantitative platform for China A-shares. The central research question is whether a large language model can identify genuine **AI compute chain** exposure inside the **CSI 300 (沪深300)** universe, and whether an equal-weighted, semi-annually rebalanced portfolio of those names can outperform a CSI 300 ETF benchmark. The repository also contains a Barra-style factor attribution engine and a separate LLM-assisted knowledge-base pipeline for point-in-time thematic research.

The system is intended for **research and backtesting only**. It is not investment advice, and it does not place orders.

---

## 1. What the Project Does

The repository implements four connected workflows:

| # | Workflow | Entry point | Purpose |
|---|----------|-------------|---------|
| 1 | LLM stock selection + multi-period backtest | `target_code_pipeline.py` | Classifies CSI 300 constituents into the AI compute chain using DeepSeek, then runs an equal-weight buy-and-hold backtest against a 510300 ETF benchmark. |
| 2 | Benchmark evaluation | `eval_hs300_perform.py` | Downloads 510300 ETF data via akshare, computes buy-and-hold performance, and writes the benchmark CSV consumed by workflow 1. |
| 3 | Barra-style factor attribution | `barra_model_analysis.py` | Decomposes active return versus a holdings-weighted CSI 300 benchmark into size, value, momentum, volatility and stock-specific contributions. |
| 4 | Knowledge-base thematic research | `knowledge_base/news_collect.py`, `knowledge_base/industry_eval.py` | Builds point-in-time technology event dossiers with web-search-enabled DeepSeek, then screens 13 industry chains with a "0→1 theme investing" four-condition framework. |

Supporting utilities acquire and normalize daily K-lines, CSI 300 constituents/weights, valuation data and company business descriptions.

---

## 2. Repository Layout

```
quant/
├── target_code_pipeline.py            # Workflow 1: selection + backtest orchestration
├── eval_hs300_perform.py              # Workflow 2: 510300 ETF benchmark report
├── barra_model_analysis.py            # Workflow 3: factor attribution
├── group_return_analysis.py           # AI vs non-AI two-group period return diagnostic
├── strategy/
│   ├── backtest_buy_hold.py           # Multi-period buy-and-hold engine (costs, benchmark, IR)
│   ├── industry_chain_classifier.py   # DeepSeek AI-compute-chain classifier
│   ├── calc_size_factor_exposure.py   # Size factor: ln(amount / turnover), z-scored
│   ├── calc_value_exposure.py         # Value factor: 1/PB, z-scored
│   ├── calc_momentum_exposure.py      # Momentum factor over a window, z-scored
│   ├── calc_volatility_exposure.py    # Realized volatility factor, z-scored
│   ├── calc_excess_return.py          # Cross-sectional return / excess-return helper
│   └── factor_return_cross_sec_regression.py  # WLS cross-sectional risk model
├── utils/
│   ├── text_utils.py                  # CSV readers, code normalization, PB query helpers
│   ├── security_utils.py              # API-key / credential redaction
│   └── plot_utils.py                  # Clustered bar and K-line plotting helpers
├── data/
│   ├── fetch_daily_k.py               # baostock daily K-line fetcher (resumable)
│   ├── fetch_hs300_constituents.py    # baostock constituent snapshots
│   ├── fetch_hs300_valuation.py       # PE/PB snapshots at half-year ends
│   ├── fetch_main_business.py         # akshare/THS business-description fetcher
│   ├── hs300_weight_YYYYh{1,2}.csv    # Required input: semi-annual CSI 300 holdings/weights
│   ├── hs300_valuation_6m.csv         # PE(TTM) / PB(MRQ) by code and target date
│   └── stock_main_business/main_business.csv
├── stock_daily_k_hfq/                 # Back-adjusted daily K cache
├── stock_daily_k_bfq/                 # Unadjusted daily K cache
├── knowledge_base/
│   ├── news_collect.py                # DeepSeek web-search event dossier generator
│   ├── news_collect_prompt.md         # Yearly event-dossier prompt template
│   ├── industry_eval.py               # Map/compress/final thematic screening pipeline
│   ├── industry_eval_prompt.md        # Four-condition screening framework
│   ├── output/                        # Yearly event dossiers and quarterly splits
│   └── analysis/                      # Evidence cache, prompts and final report
├── test/                              # pytest tests (redaction, metrics, group analysis)
├── portfolio_backtest.log             # Generated backtest log
├── portfolio_nav.png                  # Generated NAV chart
├── group_return_analysis.csv          # Generated per-period group-return detail
├── group_return_analysis.md           # Generated group-return summary and caveats
├── group_return_spread.png            # Generated group-return chart
├── factor_contrib_detail.csv          # Generated factor attribution detail
└── factor_contrib_barplot.png         # Generated factor contribution chart
```

Directories that appear at runtime but are not source code:

- `stock_daily_k_hfq/individual/*.csv`, `stock_daily_k_bfq/individual/*.csv` — per-stock K-line caches (461 stocks each in the current dataset).
- `target_code_llm_cache.jsonl` — append-only LLM classification cache.
- `tmp_target_code.csv` / `target_code.csv` — generated target holdings.
- `hs300etf_510300_<start>_<end>_performance.csv/.png` — generated benchmark reports.
- `knowledge_base/analysis/work/evidence/` — cached LLM evidence extraction.

Source code comments and docstrings are primarily in Chinese; this README is the English overview.

---

## 3. Data Conventions

### Stock codes

Codes are accepted as 6-digit strings (`000063`) or baostock-style strings (`sz.000063`, `sh.600000`, `bj.430047`). `utils.text_utils.normalize_code` converts all forms to `sh.|sz.|bj.` prefix format; per-stock cache files replace the dot with an underscore (`sz_000063.csv`).

### Daily K-line files

Each individual cache has the columns:

```
date, code, open, high, low, close, preclose, volume, amount, turn, pctChg, tradestatus, isST
```

- `stock_daily_k_hfq` — back-adjusted (`adjustflag=1`). This is the primary series for returns, momentum, volatility and size.
- `stock_daily_k_bfq` — unadjusted (`adjustflag=3`), refreshed alongside and used in data-coverage checks.
- Suspended days (`tradestatus != 1`) are removed by the fetcher.
- Files are written atomically (`.tmp` + `os.replace`) after merging new rows by date.

### CSI 300 weight files

`data/hs300_weight_YYYYh1.csv` (period end June 30) and `...h2.csv` (period end December 31) are required inputs. They contain columns including:

```
序号, 股票代码, 股票名称, 数量（股）, 公允价值（元）, 占基金资产净值比例（%）
```

`read_hs300_weight` extracts `股票代码` and `占基金资产净值比例`, zero-pads codes to 6 digits and returns a `code -> weight` table. A weight file dated `YYYYh1` is treated as the holdings snapshot for the **end** of that half-year; it becomes the **beginning** holdings for the period that starts on that date (and remains the basis for any later date until the next snapshot).

### Valuation file

`data/hs300_valuation_6m.csv`:

```
code, target_date, actual_date, peTTM, pbMRQ
```

One row per stock and half-year-end snapshot. `query_pb_mrq` supports `nearest`, `previous` and `exact` matching; attribution uses `exact` to avoid look-ahead.

### Target holdings file

`tmp_target_code.csv` (working file) and `target_code.csv` (published file):

```
query_date,target_code
2021-01-01,000063 000066 000938 ...
2021-06-30,000063 000066 ...
```

`query_date` must be unique, parseable and sorted ascending. `target_code` is a space-separated list of 6-digit codes. The first row is the first rebalance date, not the first day of available data.

### Benchmark performance file

`eval_hs300_perform.py` writes a two-section CSV:

```
# Performance Summary
Metric,Value
...
# Daily Series
,Close,Cumulative NAV,Drawdown
...
```

`strategy.backtest_buy_hold.load_benchmark_data` reads the `# Daily Series` section and uses the cumulative NAV column.

---

## 4. Workflow 1 — LLM Selection + Multi-Period Backtest

`target_code_pipeline.py` is the main entry point. It runs in two stages: **prep** (LLM selection) and **eval** (backtest).

### 4.1 Rebalance calendar and universe

The rebalance dates are fixed in `QUERY_DATES`:

```
2021-01-01,
2021-06-30, 2021-12-31,
2022-06-30, 2022-12-31,
2023-06-30, 2023-12-31,
2024-06-30, 2024-12-31,
2025-06-30, 2025-12-31,
2026-06-30
```

For each rebalance date, the pipeline uses the **most recent weight file with period end ≤ that date** as the period's beginning universe:

| Rebalance date | Beginning holdings from |
|---|---|
| 2021-01-01 | `data/hs300_weight_2020h2.csv` |
| 2021-06-30 | `data/hs300_weight_2021h1.csv` |
| 2021-12-31 | `data/hs300_weight_2021h2.csv` |
| ... | ... |

This is the "previous period-end holdings = current period beginning holdings" convention. If a query date falls between half-year ends, the latest earlier file is reused.

### 4.2 Prep stage: LLM classification

For every unique stock appearing in any rebalance-date universe:

1. The company description is assembled from `data/stock_main_business/main_business.csv` fields `主营业务`, `产品类型`, `产品名称`, `经营范围`.
2. `strategy.industry_chain_classifier.classify_ai_compute` sends the description to DeepSeek and asks for a strict JSON verdict:
   ```json
   {"is_in_chain": true, "reason": "one-line reason"}
   ```
3. The prompt defines the AI compute chain as:
   - **Upstream**: AI chips/GPU/ASIC/FPGA, HBM and storage, optical modules/chips, high-speed PCB, copper interconnect, liquid cooling, server power, switches, advanced packaging, semiconductor equipment/materials, EDA.
   - **Midstream**: AI servers, intelligent data centers/IDC, cloud compute services, networking equipment, compute leasing.
   - **Downstream**: LLM training/inference services and AI applications where compute is a core cost or selling point.
   - Traditional businesses (real estate, baijiu, banking/insurance, legacy ICE vehicles, coal, consumer, pharma, media, etc.) are classified `false`; concept-only exposure without substantive revenue tends to be classified `false`.
4. Results are cached in `target_code_llm_cache.jsonl`. Cache key = `(code, SHA-1 of description, prompt version, model)`. Any change to the description, prompt or model invalidates the entry automatically.
5. The cache is append-only and flushed after each verdict, so an interrupted run resumes from the last completed stock.
6. For each rebalance date, all universe members whose latest verdict is `true` are written as space-separated codes. An empty selection for any date aborts the pipeline.

The current dataset has 12 rebalance dates, ~3,600 holdings appearances and ~461 unique codes, so at most ~461 API calls are needed on a cold cache.

The default classifier model is `deepseek-chat` (`DEFAULT_MODEL` in `strategy/industry_chain_classifier.py`). The prompt lives in that file; `PROMPT_VERSION` is a SHA-1 prefix of the prompt text.

### 4.3 Target CSV and backtest range

- The target CSV is written atomically to `tmp_target_code.csv` by default.
- `--publish` copies the working target file to `target_code.csv` only after a successful backtest, backing up any existing file as `target_code.csv.bak-<timestamp>`. Unless `--skip-prep` is also supplied, the publish run performs selection and backtesting again first (LLM results are normally served from cache).
- The backtest start is fixed at **2021-01-01**; an earlier `--start-date` is clamped with a warning.
- The end date defaults to the run date, capped at **2026-12-31**.
- If the start lies between rebalance dates, the latest earlier period is used and the first period begins on the requested start date.
- If the end date is after the last rebalance date, the final period extends to the end date and keeps the last selection.
- Periods whose interval does not overlap the requested range are dropped.

### 4.4 Pre-backtest data validation and update

Before running:

1. **K-line coverage update** — for every held stock, the pipeline checks whether its cache covers the end of its last holding period. If not, it fetches the missing range from baostock (`adjustflag=1`, back-adjusted) and merges rows atomically. `--no-update-k` skips this step.
2. **Price file validation** — verifies that every stock has a readable file with valid dates and that its first trading day is not later than the first trading day of the holding period. Missing files, unreadable files, or stocks with no data at the period start abort the backtest rather than silently shifting the start.
3. **Backups** — existing `portfolio_backtest.log` and `portfolio_nav.png` are copied to timestamped `.bak` files.

### 4.5 Backtest engine

`strategy/backtest_buy_hold.run_backtest` implements a chained, multi-period, equal-weight buy-and-hold portfolio:

1. At each period start, allocate the period's capital equally across the selected stocks using the first common trading day's close.
2. Hold without rebalancing until the period end (the next rebalance date, or the requested end date for the final period).
3. At the period end, sell the whole portfolio and carry the net proceeds into the next period.
4. Trading costs: **50 bp one-way** (`cost_rate=0.005`), charged at both entry and exit — effectively **1% per period round trip**. Costs are reflected in the NAV on the buy and sell dates.
5. The portfolio NAV is chain-linked across periods; overlapping dates are removed so each date appears once.
6. Date alignment is `outer` for the pipeline (union of dates, forward-filled, with no backward fill), and overlapping periods are trimmed.

Reported metrics:

| Metric | Formula / meaning |
|---|---|
| Final NAV | Ending net asset value, initial = 1.0 |
| Total return | `final_nav - 1` |
| Annualized return | Geometric CAGR, 252 trading days per year |
| Annualized return (arithmetic) | Mean daily return × 252 |
| Annualized volatility | Daily return std × √252 |
| Sharpe ratio (standard) | `(mean daily return × 252 − risk_free_rate) / annualized volatility`; default `risk_free_rate = 2%`, matching `eval_hs300_perform.py` |
| Sharpe ratio (legacy CAGR/vol) | `annualized return / annualized volatility`; retained as `sharpe_cagr` for comparison |
| Max drawdown | Minimum of `NAV / running max - 1` |
| Active return (standard IR numerator) | Mean daily portfolio-minus-benchmark return × 252 (arithmetic) |
| Tracking error | Std of daily excess return × √252 |
| Information ratio (standard) | Active return / tracking error (arithmetic) |
| Geometric excess return / relative CAGR | `((P_end/P_start) / (B_end/B_start))^(1/years) − 1`; display only, not the IR numerator |
| CAGR difference | `portfolio_cagr − benchmark_cagr`; display only |

The standard Sharpe and IR use arithmetic mean returns, while the annualized return display uses geometric CAGR. The legacy CAGR/vol Sharpe is retained as `sharpe_cagr`, and the geometric excess return is retained as `geometric_excess_return` / `relative_cagr`; neither is used as the IR numerator. The attribution in Workflow 3 remains arithmetic.

The benchmark is loaded from `hs300etf_510300_*_performance.csv`. `resolve_benchmark` prefers an explicitly supplied `--benchmark` path, then the legacy `hs300etf_510300_performance.csv`, then the most recently modified matching file. Benchmark alignment uses **forward fill only**; backward fill was deliberately removed because it introduces look-ahead bias. Metrics are computed only on dates where both portfolio and benchmark are valid.

### 4.6 Logs and charts

- `portfolio_backtest.log` — full period-by-period log with holdings, entry prices, shares, daily NAV, per-stock metrics, weight drift, period overview, benchmark metrics and IR.
- `portfolio_nav.png` — portfolio NAV vs normalized 510300 ETF NAV, with alternating period shading and rebalance markers.

### 4.7 CLI reference

```bash
python target_code_pipeline.py                  # reselect and backtest (start 2021-01-01, end min(today, 2026-12-31))
python target_code_pipeline.py --skip-prep      # reuse existing --target-csv
python target_code_pipeline.py --publish        # rerun, then promote tmp_target_code.csv on success
python target_code_pipeline.py --skip-prep --publish  # promote the existing target file without reselecting
python target_code_pipeline.py --end-date 2025-12-31
python target_code_pipeline.py --no-update-k    # offline / data already complete
python target_code_pipeline.py --no-cache       # no disk cache; deduplicate within this run only
python target_code_pipeline.py --refresh-cache  # ignore existing LLM cache and re-query
python target_code_pipeline.py --benchmark path/to/benchmark.csv
python target_code_pipeline.py --risk-free-rate 0.02   # standard Sharpe rf (default 2%)
python target_code_pipeline.py --weight-glob 'data/hs300_weight_*h*.csv'
python target_code_pipeline.py --business-csv data/stock_main_business/main_business.csv
python target_code_pipeline.py --cache-file target_code_llm_cache.jsonl
```

Run from the project root; all default paths are relative to it.

### 4.8 Reliability properties

- LLM verdicts are cached by content and prompt/model version; reruns are deterministic for unchanged inputs.
- Target CSV, K-line caches and published holdings are written through temp files and atomically replaced.
- Existing logs, charts and canonical target files are backed up before overwrite.
- API exceptions are re-raised with secrets redacted via `utils.security_utils.redact_secrets`.
- The pipeline aborts rather than silently backfilling missing prices or empty selections.

### 4.9 Two-group period return diagnostic

`group_return_analysis.py` tests whether the LLM label sorts returns. For each semi-annual period it partitions the 300-name universe into the **AI group** (target codes) and the **non-AI group** (the rest), computes equal-weight gross hfq returns over the same effective boundaries as the backtest, and reports:

- per-period means, medians, cap-weighted means, the full-universe equal-weight mean, and the 510300 ETF context;
- the spread (AI − non-AI), Welch t / permutation p, bootstrap CI and coverage;
- a period-level summary (mean/median spread, t-stat on the 12 period spreads, sign test, hit rate, bootstrap CI, cumulative long-short and cumulative group growth);
- robustness windows: 2021–2024, 2025–2026 and ex-last-period.

The AI group's equal-weight mean is the gross buy-and-hold portfolio return (before costs); the non-AI group is the "rest of the index" control. The analysis is **descriptive**: the labels use current business descriptions, are not point-in-time, and the theme was chosen with hindsight. The long-short spread is a factor return, not an investable A-share strategy. Cumulative figures are products of period returns and exclude the between-period gap days (the backtest buys at the next period's first close).

```bash
python group_return_analysis.py
python group_return_analysis.py --end-date 2026-11-21
python group_return_analysis.py --no-plot
```

Outputs: `group_return_analysis.csv` (per-period detail), `group_return_analysis.md` (summary + caveats) and `group_return_spread.png` (spread bars + cumulative growth).

---

## 5. Workflow 2 — 510300 ETF Benchmark Evaluation

`eval_hs300_perform.py` downloads daily ETF data from akshare (East Money) and computes a buy-and-hold performance report.

- Default symbol: `510300` (CSI 300 ETF).
- Fixed range: **2021-01-01** to the run date, capped at **2026-12-31**.
- Default adjustment: `hfq` (back-adjusted); supports `qfq` and unadjusted.
- Suspended days (zero volume) are removed.

Metrics: trading days, total return, annualized return (geometric), annualized return (arithmetic), annualized volatility, standard arithmetic Sharpe ratio with configurable annual risk-free rate (default 2%), legacy CAGR/vol Sharpe for comparison, max drawdown and its start/end dates, Calmar ratio, daily win rate, initial/final price.

Outputs default to `hs300etf_<symbol>_<start>_<end>_performance.csv` and `.png`. The CSV contains both the summary and daily series and is the benchmark input for Workflow 1.

```bash
python eval_hs300_perform.py                    # default 510300, hfq, rf=2%
python eval_hs300_perform.py -s 159919 -a qfq -r 0.015
python eval_hs300_perform.py --no-show          # save without opening a window
python eval_hs300_perform.py --no-plot          # summary CSV only
```

---

## 6. Workflow 3 — Barra-Style Factor Attribution

`barra_model_analysis.py` explains the strategy's **active return** relative to a CSI 300 holdings benchmark using four style factors. It is a research attribution tool, not part of the live selection logic.

### 6.1 Test periods and benchmark

- Test range: **2021-01-01** to the run date, capped at **2026-12-31**.
- Rebalance boundaries: the test start date, every June 30 and December 31, and the run date.
- Attribution covers each interval `[rebalance date, next boundary]`, beginning at 2021-01-01. The first period uses a **2020-07-01 → 2021-01-01** factor window so that beginning-of-period exposures are available.
- Target holdings: `target_code.csv` if it exists, otherwise `tmp_target_code.csv`; target weights are equal-weight.
- Benchmark holdings: the most recent weight file with period end ≤ the period start (the same "previous period-end" convention as the backtest). Benchmark return is the beginning-weight-weighted return of those holdings, with any weight shortfall treated as cash earning 0%. The 510300 ETF series is **not** used here.

### 6.2 Factor exposures (point-in-time)

All exposures use only data available at the rebalance date:

| Factor | Definition | Window |
|---|---|---|
| Size | `ln(float market cap)`, where `float_mv ≈ amount / turn`; winsorized at 1%/99%, cross-sectionally z-scored | Rebalance date (or latest trading day before it) |
| Value | `1 / pbMRQ`; winsorized and z-scored | Latest valuation snapshot with `target_date ≤ rebalance date`, then an exact match on that snapshot date |
| Momentum | `last close / first close - 1` on the back-adjusted series; winsorized and z-scored | `[previous rebalance date, current rebalance date]` |
| Volatility | Standard deviation of daily returns; cross-sectionally z-scored (not annualized) | Same window as momentum |

The `amount` and `turn` fields are identical in the hfq and bfq caches; hfq is used for all factor calculations. The first period's factor window starts **2020-07-01**, and a one-time backfill fills the 2020 H2 price history (marker: `data/.price_history_2020h2.done`).

### 6.3 Risk model and return decomposition

For each period, the model estimates factor returns by cross-sectional weighted least squares:

```
r_i = Σ_k X_{i,k} · f_k + u_i
```

- `r_i` = stock return minus the holdings-weighted benchmark return (excess return).
- `X` = standardized size, value, momentum and volatility exposures.
- `u_i` = stock-specific return.
- WLS weights: `w_i = market_cap_i ** 0.5` (square-root market-cap weighting); the implementation multiplies `X` and `r` by `sqrt(w)` and runs OLS.
- Only stocks with finite exposures, finite positive market cap and finite returns enter the regression; at least `k + 1` valid stocks are required.

Attribution on the active weights `A_i = w_target,i - w_bench,i`:

```
factor contribution_k = f_k · Σ_i A_i · X_{i,k}
specific contribution   = Σ_i A_i · u_i
total active return     = Σ_i A_i · r_i
```

The script also reports the benchmark-residual term (the benchmark holdings' contribution, with any weight shortfall treated as 0%-return cash) and an unattributed residual. An assertion checks that the factor-plus-specific decomposition reconciles with the total active return to numerical precision.

There are **no market or industry factors** and no industry constraints; this is a deliberately compact style model.

### 6.4 Data updates

Before attribution the script:

1. Probes the vendor's latest available trading date using a liquid reference stock.
2. Incrementally updates stale hfq and bfq caches to `min(required end, vendor latest)`.
3. Rebuilds each `all_daily_k.csv` from the individual files (the fetcher writes only the updated subset).
4. Backfills the 2020 H2 hfq prefix once for the first period's factor window.
5. Checks per-stock coverage to each period's effective end and reports delisted/suspended stocks with no data.

Set `UPDATE_PRICE_DATA = False` in the module for offline runs. The effective period end is the earlier of the nominal end, the common last trading date of the stock pool, and the vendor's latest date. Stocks without data are excluded from the regression and reported through missing-weight and coverage diagnostics; they are not silently treated as zero returns.

### 6.5 Outputs

- `factor_contrib_detail.csv` — one row per period with effective dates, sample counts, factor contributions, specific return, benchmark residual, unattributed return, total active return, reconciliation check, coverage diagnostics, benchmark return and target-only holdings.
- `factor_contrib_barplot.png` — clustered bars per period for size, value, momentum, volatility, specific and total contributions.

The console log prints the same decomposition and a label such as `2021h1`; a `*` suffix marks a period whose effective end was truncated before the nominal period end.

```bash
python barra_model_analysis.py
```

---

## 7. Workflow 4 — Knowledge-Base Thematic Research

This is an independent research pipeline for building a **point-in-time** evidence base and screening industry chains. It does not feed the portfolio backtest automatically; its output is an analyst-style Markdown report.

### 7.1 Event dossier collection (`knowledge_base/news_collect.py`)

The collector sends `news_collect_prompt.md` to DeepSeek with the `deepseek_web_search` tool enabled. The prompt template supports `{year}` and `{next_year}` placeholders.

Key design choices:

- Default model `deepseek-flash`; web search supports normal search or "deep thinking search".
- Because responses that used the search tool cannot be continued, web-search runs are split into **four independent quarterly requests (Q1–Q4)** by default.
- Each quarter is streamed to its own file; a combined Markdown file is written alongside. Per-quarter failure, truncation (`finish_reason=length`) or content filtering is reported and produces a non-zero exit code.
- Search evidence (queries, source URLs, citation counts, tool calls) is verified and printed to stderr; `--verify-search` runs a small probe request to confirm that server-side search actually executed.
- The prompt enforces a strict as-of view: only facts, policies and valuations visible by **December 31 of the target year**; no later outcomes, no revised data, no post-hoc interpretation. Every event needs a verifiable primary source and a date; unsupported items are excluded or marked as unverified.

The generated dossiers contain a yearly technology storyline, event cards, policy ledger and industry-chain valuation snapshots.

Current `knowledge_base/output/` contains dossiers for **2015–2020**, with annual merged files (`YYYY_events.md`) and quarterly splits (`YYYY_events_Q1..Q4.md`).

Example:

```bash
python knowledge_base/news_collect.py --year 2021 \
    --output knowledge_base/output/2021_events.md
python knowledge_base/news_collect.py --year 2021 --no-search --no-quarters
python knowledge_base/news_collect.py --verify-search
```

### 7.2 Four-condition thematic screening (`knowledge_base/industry_eval.py`)

The evaluator reads `industry_eval_prompt.md` and applies a "0→1 theme investing" framework to the event dossiers. The framework parses to 13 industry chains and four conditions:

**Chains:** new energy vehicles; wind/solar/storage and new power systems; AI compute/semiconductors/data centers; defense technology; biotech; industrial software and smart manufacturing; metaverse and VR/AR; IT localization and digital economy; commercial space; traditional energy and resources; seeds and breeding; internet; real estate.

**Conditions:**

1. **Technology inflection** — a step change in performance, cost, process or product form with a path to industrialization; lab concepts and one-off news do not count.
2. **Policy backing** — formally issued plans, subsidies, tax, access, procurement, standards, regulation or export controls with substantive support or restriction; statements and drafts do not count.
3. **Not yet commercialized** — no large-scale commercial revenue yet, or penetration/shipments still early; mass-market price wars disqualify.
4. **Large addressable space** — market size, penetration headroom or quantified policy targets large enough to support a major theme; unquantified claims are doubtful.

Pipeline stages:

1. **Discover and chunk** — default files are annual `*_events.md` dossiers (quarterly splits are excluded to avoid duplicate evidence). Documents are split on `##`/`###` headings, with hard character splitting and overlap if needed.
2. **Map (evidence extraction)** — each chunk is sent to the model concurrently (`--workers`, default 3) with a strict extraction-only prompt. Evidence keeps its date and card/policy ID, distinguishes facts from material inference, and never adds outside knowledge. Results are cached in `analysis/work/evidence/` with a sidecar `.json` keyed by prompt version, model, thinking mode, framework, file and chunk.
3. **Regroup** — evidence is reorganized by industry chain across years; unmatched bullets go to an "unclassified evidence" section.
4. **Compress** — if total evidence exceeds `--max-evidence-chars` (default 60,000), each chain is compressed to a per-chain budget with a separate cached LLM call; failures fall back to the original text.
5. **Final report** — the framework plus evidence is sent to the model with a 32,768-token output budget. The report must contain a conclusion table grouped **preferred → watch → eliminated**, a deep-dive for preferred chains (key 0→1 variables, catalyst windows, risks), notes on watch/eliminated chains, and an evidence-boundary section. A post-processing pass reorders the table if the model did not follow the required grouping.

Design constraints: no look-ahead evidence, no unsupported facts, every evidence item traceable to a date and ID, and partial failures do not silently disappear (`--allow-partial` must be explicit).

```bash
python knowledge_base/industry_eval.py --dry-run   # build prompts and run plan, no API calls
python knowledge_base/industry_eval.py             # full map → compress → report
python knowledge_base/industry_eval.py --refresh   # ignore evidence cache
python knowledge_base/industry_eval.py --input knowledge_base/output/2021_events.md
```

Outputs:

- `knowledge_base/analysis/industry_eval_report.md` — final Markdown report with a header recording evidence cutoff, model, source files and generation time.
- `knowledge_base/analysis/work/prompts/` — exact map and final prompts.
- `knowledge_base/analysis/work/evidence/` — cached evidence blocks and sidecar metadata.
- `knowledge_base/analysis/work/run_plan.json` — dry-run plan (chains, conditions, files, chunk counts, as-of date).

The current generated report uses evidence through **2020-12-31**; its conclusion table ranks the AI compute/semiconductor/data-center chain as the only chain with all four conditions met.

---

## 8. Data Acquisition Utilities

| Script | Source | Output | Notes |
|---|---|---|---|
| `data/fetch_daily_k.py` | baostock | `all_daily_k.csv`, `all_daily_k.parquet`, `individual/*.csv` | Resumable: existing individual files are reused. Supports adjust flags 1/2/3. The `__main__` block is a small demo; production callers pass `codes`, `start_date`, `end_date`, `out_dir`. |
| `data/fetch_hs300_constituents.py` | baostock `query_hs300_stocks` | `data/hs300_constituents_2021_2026.csv` | Snapshots on 2021-01-01 plus each Jun 30/Dec 31. Optional; the attribution uses weight files instead. Run from `data/` because the output path is relative. |
| `data/fetch_hs300_valuation.py` | baostock daily PE/PB | `data/hs300_valuation_6m.csv` | Uses all weight files' code union; incremental by default, `--full` to refetch. Queries a 90-day lookback and keeps the last row on/before each target date. |
| `data/fetch_main_business.py` | akshare (THS) | `data/stock_main_business/main_business.csv` | Resumable and batched. `OUTPUT_DIR` is `./stock_main_business`, so run it from `data/` (or adjust the constant) so the pipeline finds `data/stock_main_business/main_business.csv`. |

Weight files themselves have no downloader in this repository and must be supplied externally.

---

## 9. Setup

### Requirements

- Python **3.10+** (the code uses `X | None` and built-in generic annotations; tested on Python 3.13).
- Packages:

```bash
pip install pandas numpy matplotlib seaborn openai python-dotenv baostock akshare pytest
# optional: pyarrow (Parquet output from fetch_daily_k)
# optional: mplfinance (only for utils.plot_utils.plot_k_line)
```

### Environment file

Create `.env` in the project root:

```dotenv
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`DEEPSEEK_API_KEY` is required for Workflows 1 and 4. `.env` is loaded from the project root by both the classifier and the knowledge-base scripts. Never commit this file; all errors and logs pass through `utils.security_utils.redact_secrets` to avoid printing keys or credentialed URLs.

### Offline mode

- Workflow 1: `--no-update-k`, combined with `--skip-prep` or warm caches.
- Workflow 3: set `UPDATE_PRICE_DATA = False` in `barra_model_analysis.py`.
- Workflow 4: `--dry-run` builds prompts and plans without API calls; cached evidence can be reused.

---

## 10. Quick Start

Run the commands from the repository root unless a step says otherwise.

```bash
# 0) Install dependencies and configure .env (see above)

# 1) Refresh business descriptions (run from data/ so output lands where the pipeline expects)
cd data && python fetch_main_business.py && cd ..

# 2) Refresh PE/PB valuation snapshots for all half-year universes
python data/fetch_hs300_valuation.py

# 3) Refresh the 510300 benchmark report (also validates akshare access)
python eval_hs300_perform.py --no-show

# 4) Select AI-compute-chain stocks and run the multi-period backtest
python target_code_pipeline.py

# 5) Optionally publish the working target file without reselecting
python target_code_pipeline.py --skip-prep --publish

# 6) Attribute active return to style factors
python barra_model_analysis.py

# 7) Build/refresh the thematic research dossier, then evaluate
python knowledge_base/news_collect.py --year 2021 \
    --output knowledge_base/output/2021_events.md
python knowledge_base/industry_eval.py --dry-run
python knowledge_base/industry_eval.py
```

Daily K-line caches are created and incrementally extended automatically by Workflows 1 and 3; no separate full-universe download is required.

### Tests

```bash
pytest
```

The current test suite covers credential redaction (`test/test_security_utils.py`), metric conventions (`test/test_backtest_metrics.py`) and the two-group analysis (`test/test_group_return_analysis.py`).

---

## 11. Generated Artifacts

| File | Producer | Content |
|---|---|---|
| `tmp_target_code.csv` | `target_code_pipeline.py` | Working selections by rebalance date |
| `target_code.csv` | `target_code_pipeline.py --publish` | Published selections |
| `target_code_llm_cache.jsonl` | `target_code_pipeline.py` | Append-only LLM verdict cache with code, description hash, prompt version, model, verdict, reason, timestamp |
| `portfolio_backtest.log` | `backtest_buy_hold.run_backtest` | Full backtest log and performance summary |
| `portfolio_nav.png` | `backtest_buy_hold.plot_nav` | Portfolio vs benchmark NAV |
| `hs300etf_510300_*_performance.csv/.png` | `eval_hs300_perform.py` | Benchmark summary, daily NAV/drawdown, chart |
| `group_return_analysis.csv` | `group_return_analysis.py` | Per-period AI/non-AI group returns, spread, inference and coverage |
| `group_return_analysis.md` | `group_return_analysis.py` | Group-return summary table and caveats |
| `group_return_spread.png` | `group_return_analysis.py` | Per-period spread bars and cumulative group growth |
| `factor_contrib_detail.csv` | `barra_model_analysis.py` | Per-period factor and specific attribution |
| `factor_contrib_barplot.png` | `barra_model_analysis.py` | Factor contribution bars |
| `knowledge_base/output/*_events.md` | `news_collect.py` | Yearly event cards, policy ledger, valuation snapshots |
| `knowledge_base/analysis/industry_eval_report.md` | `industry_eval.py` | Four-condition screening report |
| `knowledge_base/analysis/work/**` | `industry_eval.py` | Prompts, evidence cache, run plan |

Existing logs, charts and canonical target files are backed up with `.bak-<timestamp>` suffixes before being overwritten.

---

## 12. Design Principles and Known Limitations

### Principles

- **Point-in-time correctness** — no backward-filled benchmark values; valuation lookups are exact/as-of; constituents and weights come from the previous period-end file; the knowledge-base prompt forbids post-cutoff facts.
- **Reproducibility** — content-addressed LLM caches, explicit prompt versions, atomic writes, deterministic backtests for unchanged inputs.
- **Metric conventions** — return display uses geometric CAGR; Sharpe and IR use arithmetic mean excess returns; the legacy CAGR/vol Sharpe (`sharpe_cagr`) and geometric excess return (`geometric_excess_return` / `relative_cagr`) are retained for comparison but are not the IR numerator.
- **Explicit failure** — missing prices, empty selections, invalid targets and failed K-line updates abort rather than silently degrading.
- **Secret hygiene** — API keys and credentialed URLs are redacted from all surfaced errors; `.env` is used for configuration.

### Limitations

- **Simplified cost model** — a flat 50 bp per side; no slippage, stamp duty, commission minimums, market impact or borrow costs.
- **Equal weights** — target weights are equal, not optimized. In the attribution, benchmark weights come from provided half-year snapshots and may sum to less than 100%; the remainder earns 0% cash.
- **Compact risk model** — no market, industry, or macro factors; size is approximated by `amount / turn`; volatility and momentum use simple window statistics. The attribution is explanatory, not a production risk model.
- **Data dependency** — baostock and akshare can lag, rate-limit, or lack delisted/suspended coverage; the code reports these cases but cannot eliminate them.
- **Universe construction** — the current universe comes from half-year weight snapshots rather than a daily point-in-time index membership history, so mid-period index changes are not modeled.
- **LLM dependency** — selection and dossier generation require DeepSeek API access; results inherit model variability. Verdicts are cached, and prompt/model changes invalidate the cache.
- **Knowledge-base scope** — the current dossiers cover 2015–2020 and the report is explicitly an as-of-2020-12-31 information set; later years must be collected and evaluated separately.
- **Research only** — no order execution, portfolio accounting, compliance checks or live risk controls.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| AI compute chain | The upstream/midstream/downstream supply chain serving AI training and inference compute |
| hfq / bfq | Back-adjusted / unadjusted price series |
| Rebalance date | Semi-annual date at which the target portfolio is reselected and re-established |
| CAGR | Compound annual growth rate; geometric annualized return |
| Active return | Mean daily portfolio-minus-benchmark return (arithmetic), annualized by × 252 for the standard IR |
| Tracking error | Annualized standard deviation of daily arithmetic active returns |
| Information ratio | Standard arithmetic active return divided by tracking error |
| Geometric excess return / relative CAGR | Full-period annualized relative return, `((P_end/P_start)/(B_end/B_start))^(1/years) − 1`; display only |
| Sharpe ratio (standard) | Arithmetic annualized excess return divided by annualized volatility |
| Sharpe ratio (legacy CAGR/vol) | Geometric annualized return divided by annualized volatility; retained as `sharpe_cagr` |
| 0→1 theme | A theme transitioning from technological possibility to commercialization |
| As-of / point-in-time | Using only information that was publicly available at the stated date |
