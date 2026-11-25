# Read-only research dashboard

A local Streamlit dashboard for the quant research artifacts in this repository.
The UI is in English; source content such as stock names and LLM reasons remains
in Chinese.

**Scope:** read-only, local-only, Streamlit.

- It never calls the DeepSeek API.
- It never runs a backtest or refreshes market data.
- It never writes to research artifacts.
- It binds to `127.0.0.1` and has no login or multi-user support.

## Pages

| Page | What it shows |
|---|---|
| Overview | NAV vs benchmark, drawdown, period returns, headline metrics |
| Holdings & periods | Per-period holdings, entry prices, contributions, stock K-line |
| Factor attribution | Size / value / momentum / volatility / specific contributions |
| AI vs non-AI | Per-period group returns, spreads, bootstrap CI, p-values |
| Knowledge & selection | Cached LLM verdicts, rebalance selections, research reports |
| Data & methodology | Provenance, coverage, metric definitions, limitations |

## Data flow

```text
Research scripts
  ├─ target_code_pipeline.py --export-details ──▶ web/data/backtest_details.json
  ├─ barra_model_analysis.py ───────────────────▶ factor_contrib_detail.csv
  ├─ group_return_analysis.py ──────────────────▶ group_return_analysis.csv
  ├─ target_code_llm_cache.jsonl
  └─ knowledge_base/**/*.md
                         │
                         ▼
                web/export_data.py
                         │
                         ▼
              web/data/*.json (gitignored)
                         │
                         ▼
                 web/app.py (Streamlit)
```

`web/export_data.py` is the only data contract. The app reads JSON only; it does
not parse the backtest log or recompute research metrics.

## Quick start

From the project root:

```bash
pip install -r web/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1) Write structured backtest details (reuse existing selection and price caches)
python target_code_pipeline.py --skip-prep --no-update-k \
    --export-details web/data/backtest_details.json

# 2) Build the dashboard JSON
python web/export_data.py

# 3) Start the local app
web/run.sh
```

Open http://localhost:8502. From a Windows browser under WSL2, use `localhost`
rather than `127.0.0.1`; Windows may already own the loopback port.

`web/refresh.sh` runs steps 1 and 2. It does not call the LLM and does not
refresh market data. Remove `--skip-prep` or `--no-update-k` if you want the
full research workflow.

## Refresh routine

After changing any research artifact:

1. Re-run the relevant research script (pipeline, attribution, group analysis,
   or knowledge base).
2. Re-run `python web/export_data.py`.
3. Click **Reload data** in the sidebar, or restart the app.

A full refresh sequence:

```bash
python target_code_pipeline.py --skip-prep --no-update-k \
    --export-details web/data/backtest_details.json
python barra_model_analysis.py
python group_return_analysis.py
python web/export_data.py
web/run.sh
```

## Security notes

- `web/.streamlit/config.toml` binds the server to `127.0.0.1`.
- The app does not read the project's `.env` file.
- Generated JSON lives in `web/data/` and is gitignored.
- Do not change the bind address to `0.0.0.0` without adding authentication and
  HTTPS; the research artifacts and LLM cache are not public data.

## Tests

```bash
python -m pytest test/test_web_export.py -q
python -m pytest test/test_web_app_smoke.py -q
```

The smoke test is skipped when `web/data/portfolio.json` has not been generated.
