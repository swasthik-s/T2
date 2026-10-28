# T2 — Self-Improving Trading Research System

T2 is a modular, research-first trading system following the eight-step roadmap. It is designed to investigate chart-based trading hypotheses, validate strategies with deterministic Python, and learn from documented experiments. **All broker execution is paper-only by default.**

## Principles

- **No VWAP:** VWAP is explicitly excluded from strategy features, rules, and examples.
- **Deterministic validation:** Indicators, signals, position sizing, backtests, and risk checks are computed by Python—not guessed by an LLM.
- **AI is an assistant, not an authority:** Vision/reasoning models may describe chart screenshots, suggest hypotheses, and explain failures. Suggestions must become explicit rules and pass reproducible tests.
- **No automatic strategy promotion:** A candidate cannot replace the approved strategy unless it passes the same pre-defined validation gates.
- **Paper trading only:** No live-trading adapter is included in this foundation. Risk controls must not be bypassed by an AI agent.
- **Secrets stay out of Git:** Never commit API keys, passwords, tokens, screenshots containing account information, or broker credentials.

## Eight-step roadmap

1. **System architecture** — define modules and interfaces.
2. **Data and research engine** — ingest historical/live market data, validate timestamps and OHLCV, derive features, and store experiments.
3. **Strategy development** — translate hypotheses into unambiguous, versioned rules.
4. **Backtesting and validation** — use chronological splits, realistic costs, out-of-sample tests, walk-forward analysis, and robustness checks.
5. **Controlled self-improvement** — log, diagnose, hypothesize, generate a candidate, test, compare, and promote or revert.
6. **Execution and risk management** — paper-only broker adapter, pre-trade gates, exposure and loss limits, order reconciliation, and circuit breaker.
7. **Deployment and monitoring** — scheduled jobs, health checks, alerts, persistent logs, and mobile-friendly status views.
8. **Tools and setup** — reproducible environment, repository/version control, configuration, automated tests, and deployment documentation.

## Research agents / skills

- **Market analyst:** classify market regime and describe chart observations.
- **Strategy researcher:** propose testable hypotheses.
- **Code agent:** implement approved specifications as testable Python.
- **Research reviewer:** inspect logic, bias, test quality, and robustness.
- **Execution/risk engine:** enforce deterministic limits independently of AI.

A coordinating orchestrator may call these components, but does not get permission to override risk controls.

## Live chart observation

The vision module is intended to ingest timestamped screenshots from an explicitly configured chart source. Image interpretations must be stored with instrument, timeframe, screenshot time, and the corresponding market-data window. Vision output is treated as an observation—not a trading signal—until reconciled with data and tested. T2 does not silently record or control a user's device screen.

## Current development stage

This repository starts with architecture, configuration, tests, and research scaffolding. It is **not yet a finished strategy, a continuously running service, or a proven profitable system**. Market-data and broker integrations must be configured and tested separately.

## Suggested development sequence

1. Build data validation and feature calculations.
2. Add a deterministic baseline strategy and unit tests.
3. Build the backtest engine and realistic cost model.
4. Add experiment tracking and validation gates.
5. Add optional vision/LLM analysis behind a replaceable interface.
6. Connect Alpaca paper trading only after validation and risk tests.
7. Add persistent cloud scheduling and monitoring.

## Quick start

Use Python 3.11 or later.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\\Scripts\\activate
pip install -r requirements.txt
pytest
```

The initial tests validate project safety defaults and configuration. No API key is needed to run them.

## Configuration and credentials

Copy `.env.example` to `.env` only in a trusted runtime, then populate secrets directly in that environment's secret manager. Do not upload `.env` or paste secret values into chat. The paper-trading guard must remain enabled.

## Safety notice

This is software research, not financial advice. Backtests can be misleading and do not guarantee future results. T2 must remain paper-only until the system has passed extended paper testing and independent review.
