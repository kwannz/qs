Rust FFI Backtest Integration Plan

Scope
- Add an optional Rust-powered backtest engine via PyO3 FFI.
- Keep Python engine as default; toggle Rust via env `FT_USE_RUST_FFI_BACKTEST=1`.
- Phase 1 PoC supports long-only enter/exit signals; no ROI/SL/TSL/protections.

Repository Changes
- `rust/ft_rust_backtest`: Minimal PyO3 crate exposing `simulate_trades(data, start_ts, end_ts)`.
- `freqtrade/optimize/rustffi.py`: Python wrapper to prepare data, call Rust, and normalize results.
- `freqtrade/optimize/backtesting.py`: Engine switch (env-gated) to call Rust path.

Build & Run
- Prereqs: Rust toolchain (1.74+), Python 3.11, maturin.
- Install maturin: `pip install maturin` (inside venv).
- Build module: `cd rust/ft_rust_backtest && maturin develop --release`.
- Run with Rust engine:
  - `export FT_USE_RUST_FFI_BACKTEST=1`
  - `freqtrade backtesting -c user_data/config.json --timeframe 5m`.

Data Contract (Phase 1)
- Input from Python: dict `{pair -> list[row]}` where row matches Freqtrade HEADERS:
  - `[date_ts_sec, open, high, low, close, enter_long, exit_long, enter_short, exit_short, enter_tag, exit_tag]`.
- Output from Rust: list[dict] aligned to `BT_DATA_COLUMNS` for direct DataFrame construction.

Phase Rollout
- Phase 1: Long-only PoC, simple enter/exit signals.
- Phase 2: Fees + slippage + minimal ROI + stoploss (implemented), exit_reason tagging.
- Phase 3: Detail timeframe, protections, partial exits, futures funding fees.
- Phase 4: Parity tests vs Python engine; config/schema flag instead of env; perf benchmarks.

Parameters
- Fee: Uses `self.fee` worst-case from exchange by default (can override via config `fee`).
- Stoploss: Uses `stoploss` from config/strategy (negative ratio).
- ROI: Uses `minimal_roi` dict (e.g., `{ "0": 0.04, "30": 0.02 }`).
- Slippage: Optional `backtest_slippage` (fraction, e.g., `0.0005`). Entry adds slippage, exit subtracts.

Parity & Testing
- Unit-level core alignment:
  - `pytest -q tests/optimize/test_rustffi_unit.py::test_rustffi_core_alignment`
  - Compares Rust engine results to a Python reference on a synthetic dataset（ROI/SL/滑点/费用）.
- Integration smoke:
  - Enable FFI and run `backtesting` on a small window to validate end-to-end path.

Notes
- The Rust crate is ABI3 for Python 3.11 (`abi3-py311`). Adjust if Python version differs.
- On import failure, Python falls back to the original engine and logs the reason.
