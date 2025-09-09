Backtest Slippage (Config)

Overview
- `backtest_slippage` adds a simple slippage model to backtests.
- Interpreted as a fraction. Example: `0.0005` equals 5 bps (0.05%).
- Applied asymmetrically for realism:
  - Entry: increases price by `+slippage` (worse fill)
  - Exit: decreases price by `-slippage` (worse fill)

Usage
- Add to your `config.json` or strategy config:

```json
{
  "backtest_slippage": 0.0005
}
```

Notes
- With the Rust FFI backtest engine enabled (`FT_USE_RUST_FFI_BACKTEST=1`), slippage is applied inside the engine and reflected in net PnL.
- If not set, slippage defaults to `0.0`.

