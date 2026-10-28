# T2 self-correcting research engine (first implementation)

## What is now in the repository

- `t2/improvement/gpu_ml.py`: causal OHLC-derived features, forward-direction labels, optional PyTorch MLP training, automatic CUDA/MPS/CPU selection, and probability inference.
- `scripts/run_gpu_ml_experiment.py`: trains only on the first 60% of the timeline, purges training labels whose forward horizon crosses the boundary, chooses a probability threshold on the next 20%, then evaluates only that selected threshold on the final 20% test segment. Reports cost sensitivity and the selected compute device.
- `t2/improvement/feedback.py`: deterministic feedback rules to classify validation failures, held-out failures, and cost-sensitive results and recommend the next research direction. These rules do not edit source code or promote strategies.
- `tests/test_gpu_ml_and_feedback.py`: causal-feature, label-boundary, feedback, and safety invariant tests.

## Run in Colab

Pull the code and run the tests first:

```python
%cd /content/T2
!git pull origin main
!python -m pytest -v
```

Check whether the runtime exposes a GPU:

```python
!nvidia-smi
!python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

Then run the ML experiment:

```python
!python scripts/run_gpu_ml_experiment.py
```

A GPU runtime alone is not enough: PyTorch must be installed in a build compatible with the runtime. If PyTorch is missing, install the appropriate build for the current Colab runtime and restart the session. The runner will fall back to CPU when PyTorch exists but CUDA is unavailable; it raises a clear error if PyTorch is not installed.

## Meaning of “self-correcting”

The engine can use recorded results to reject failed candidates, diagnose likely failure modes, and choose the next bounded experiment. It must not rewrite its own source, relax risk controls, or promote a strategy just because it found a high score. A future orchestrator can call these feedback rules after each run and queue the next experiment family.

## Important limitations

- The MLP is an experimental **future-close-direction classifier**, not a predictor of target-before-stop probability. A later version should label outcomes using the exact trade simulator and purge samples around all split boundaries.
- The model currently uses OHLC-derived features and UTC session features; it does not use volume, VWAP, chart images, news, or order-book data.
- Source data is bid-only. Spread/commission assumptions are provisional, and PnL is in the backtest's account-currency units.
- The ML experiment is long-only, one model fit, and one chronological split. Passing one split is not evidence of durable edge; use rolling walk-forward validation, multiple regimes, cost calibration, and paper trading before any strategy can be considered.
- GPU acceleration applies to the neural-network training/inference, not the existing event-driven pandas backtester. A GPU is not guaranteed to be faster for this small MLP.
- No broker integration, order submission, live execution, automatic promotion, or automatic source-code modification is enabled.
- Current dataset lacks source volume. Do not calculate or claim VWAP results from fabricated volume.

## Next stages

1. Run this baseline ML experiment and compare its validation/test performance with the EMA baseline using identical data and costs.
2. Add ORB and EMA pullback/crossover strategy adapters to the same registry.
3. Add VWAP only after obtaining genuine volume/tick-volume data and defining session reset semantics.
4. Add a walk-forward controller, regime diagnostics, durable experiment memory, and automatic experiment scheduling with hard safety gates.
5. Consider GPU hyperparameter search only after a credible feature/label/backtest design is established.
