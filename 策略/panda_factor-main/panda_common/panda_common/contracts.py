from __future__ import annotations

from typing import Dict, Any

from .config import get_config


def _deep_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)


def get_futures_defaults() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("FUTURES_DEFAULTS", {})


def get_futures_contract(symbol: str) -> Dict[str, Any]:
    """
    返回统一维护的期货/永续合约参数（已合并默认值）。
    优先匹配精确 symbol（如 BINANCE:BTCUSDT 或 BINANCE:BTCUSDT-PERP）。
    """
    cfg = get_config()
    defaults = cfg.get("FUTURES_DEFAULTS", {})
    table = cfg.get("FUTURES_CONTRACTS", {})
    custom = table.get(symbol, {})
    merged = {**defaults, **custom}
    # 字段标准化：确保类型
    for k in ("contract_multiplier", "trade_unit", "tick_size", "min_notional",
              "maker_bps", "taker_bps", "slippage_bps", "funding_rate", "im", "mm"):
        if k in merged and merged[k] is not None:
            try:
                merged[k] = float(merged[k])
            except Exception:
                pass
    if "leverage" in merged and merged["leverage"] is not None:
        try:
            merged["leverage"] = float(merged["leverage"])
        except Exception:
            pass
    return merged

