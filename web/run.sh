#!/usr/bin/env bash
# Start the read-only dashboard on 127.0.0.1:8502.
# The address, port and theme are defined in web/.streamlit/config.toml.
set -euo pipefail
cd "$(dirname "$0")"
exec streamlit run app.py
