import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from subprocess import run, PIPE

import pytest


def _latest_result_zip(d: Path) -> Path:
    zips = sorted(d.glob("backtest-result-*.zip"))
    assert zips, f"No result zip in {d}"
    return zips[-1]


def _read_results_from_zip(z: Path) -> dict:
    with zipfile.ZipFile(z) as zf:
        # Find main JSON
        cands = [n for n in zf.namelist() if re.search(r"backtest-result-.*\\.json$", n)]
        assert cands, f"No results json in {z}"
        data = json.loads(zf.read(cands[0]).decode("utf-8"))
    return data


@pytest.mark.skipif(
    os.environ.get("FT_E2E_RUST_ALIGN") != "1",
    reason="Set FT_E2E_RUST_ALIGN=1 to run end-to-end alignment test.",
)
def test_rustffi_e2e_align(tmp_path: Path):
    """
    End-to-end check: compare Python vs Rust engines on a small window using freqtrade CLI.
    Requires user_data/config.json and a small dataset to be present locally.
    Enable via FT_E2E_RUST_ALIGN=1 to run locally (skipped on CI by default).
    """
    # Use existing config if present
    config = Path("user_data/config.json")
    assert config.exists(), "user_data/config.json not found"

    # Common args
    args = [
        "freqtrade",
        "backtesting",
        "-c",
        str(config),
        "-s",
        "SampleStrategy",
        "--timerange",
        os.environ.get("FT_E2E_TIMERANGE", "20250905-20250907"),
        "--export",
        "trades",
    ]

    # Run Python engine
    env = os.environ.copy()
    env.pop("FT_USE_RUST_FFI_BACKTEST", None)
    out_dir_py = tmp_path / "py"
    out_dir_py.mkdir(parents=True, exist_ok=True)
    rc = run(args + ["--export-directory", str(out_dir_py)], env=env, stdout=PIPE, stderr=PIPE)
    assert rc.returncode == 0, rc.stderr.decode()

    # Run Rust engine
    env_rust = env.copy()
    env_rust["FT_USE_RUST_FFI_BACKTEST"] = "1"
    out_dir_rs = tmp_path / "rs"
    out_dir_rs.mkdir(parents=True, exist_ok=True)
    rc = run(args + ["--export-directory", str(out_dir_rs)], env=env_rust, stdout=PIPE, stderr=PIPE)
    assert rc.returncode == 0, rc.stderr.decode()

    # Compare results
    z_py = _latest_result_zip(out_dir_py)
    z_rs = _latest_result_zip(out_dir_rs)
    r_py = _read_results_from_zip(z_py)
    r_rs = _read_results_from_zip(z_rs)

    # Compare trades count and total absolute profit across strategies
    def summary(d):
        strat = next(iter(d["strategy"].values()))
        return strat["results_per_pair"][-1]["trades"], strat["results_per_pair"][-1]["profit_total_abs"]

    t_py, p_py = summary(r_py)
    t_rs, p_rs = summary(r_rs)
    assert t_py == t_rs
    assert abs(p_py - p_rs) < 1e-6

