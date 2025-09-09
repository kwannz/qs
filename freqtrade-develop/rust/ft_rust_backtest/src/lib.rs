use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicBool, Ordering};

#[derive(Clone, Debug)]
struct RoiSchedule {
    items: Vec<(i64, f64)>,
}

impl RoiSchedule {
    fn from_py_dict(d: Option<&PyDict>) -> Self {
        if let Some(map) = d {
            let mut items: Vec<(i64, f64)> = Vec::new();
            for (k, v) in map.iter() {
                let m: i64 = k.extract().unwrap_or(0i64);
                let r: f64 = v.extract().unwrap_or(0.0f64);
                items.push((m, r));
            }
            items.sort_by_key(|(m, _)| *m);
            RoiSchedule { items }
        } else {
            RoiSchedule { items: Vec::new() }
        }
    }

    fn target_for_minutes(&self, elapsed_min: i64) -> Option<f64> {
        let mut ret: Option<f64> = None;
        for (m, r) in &self.items {
            if *m <= elapsed_min {
                ret = Some(*r);
            } else {
                break;
            }
        }
        ret
    }
}

fn trade_dict<'py>(
    py: Python<'py>,
    pair: &str,
    open_ts: i64,
    close_ts: i64,
    open_rate: f64,
    close_rate: f64,
    amount: f64,
    profit_abs: f64,
    profit_ratio: f64,
    enter_tag: Option<&str>,
    fee_open_abs: f64,
    fee_close_abs: f64,
) -> PyResult<&'py PyDict> {
    let d = PyDict::new(py);
    d.set_item("pair", pair)?;
    d.set_item("stake_amount", amount * open_rate)?;
    d.set_item("max_stake_amount", amount * open_rate)?;
    d.set_item("amount", amount)?;
    d.set_item("open_date", open_ts)?;
    d.set_item("close_date", close_ts)?;
    d.set_item("open_rate", open_rate)?;
    d.set_item("close_rate", close_rate)?;
    d.set_item("fee_open", fee_open_abs)?;
    d.set_item("fee_close", fee_close_abs)?;
    let duration_min = ((close_ts - open_ts) / 60) as i64;
    d.set_item("trade_duration", duration_min)?;
    d.set_item("profit_ratio", profit_ratio)?;
    d.set_item("profit_abs", profit_abs)?;
    d.set_item("exit_reason", "exit_signal")?;
    d.set_item("initial_stop_loss_abs", 0.0)?;
    d.set_item("initial_stop_loss_ratio", 0.0)?;
    d.set_item("stop_loss_abs", 0.0)?;
    d.set_item("stop_loss_ratio", 0.0)?;
    d.set_item("min_rate", open_rate.min(close_rate))?;
    d.set_item("max_rate", open_rate.max(close_rate))?;
    d.set_item("is_open", false)?;
    if let Some(tag) = enter_tag { d.set_item("enter_tag", tag)?; } else { d.set_item("enter_tag", "")?; }
    d.set_item("leverage", 1.0)?;
    d.set_item("is_short", false)?;
    d.set_item("open_timestamp", open_ts * 1000)?;
    d.set_item("close_timestamp", close_ts * 1000)?;
    let orders = PyList::empty(py);
    d.set_item("orders", orders)?;
    d.set_item("funding_fees", 0.0)?;
    Ok(d)
}

#[pyfunction]
fn simulate_trades<'py>(
    py: Python<'py>,
    data: &PyDict,
    start_ts: i64,
    end_ts: i64,
    params: Option<&PyDict>,
) -> PyResult<&'py PyList> {
    let out = PyList::empty(py);
    // Parameters
    let mut fee: f64 = 0.0;
    let mut slippage: f64 = 0.0;
    let mut stoploss: f64 = -1.0;
    let mut roi = RoiSchedule { items: Vec::new() };
    if let Some(p) = params {
        if let Ok(Some(f)) = p.get_item("fee") { fee = f.extract().unwrap_or(0.0); }
        if let Ok(Some(s)) = p.get_item("slippage") { slippage = s.extract().unwrap_or(0.0); }
        if let Ok(Some(sl)) = p.get_item("stoploss") { stoploss = sl.extract().unwrap_or(-1.0); }
        let roi_dict: Option<&PyDict> = match p.get_item("minimal_roi") {
            Ok(Some(x)) => x.downcast::<PyDict>().ok(),
            _ => None,
        };
        roi = RoiSchedule::from_py_dict(roi_dict);
    }
    let mut can_short: bool = false;
    let mut cooldown_minutes: i64 = 0;
    let mut max_drawdown_pct: f64 = 0.0;
    // Funding maps
    let mut funding_map: HashMap<String, Vec<(i64, f64)>> = HashMap::new();
    let mut mark_map: HashMap<String, Vec<(i64, f64)>> = HashMap::new();
    if let Some(p) = params {
        if let Ok(Some(cs)) = p.get_item("can_short") { can_short = cs.extract().unwrap_or(false); }
        if let Ok(Some(cd)) = p.get_item("cooldown_minutes") { cooldown_minutes = cd.extract().unwrap_or(0); }
        if let Ok(Some(mdd)) = p.get_item("max_drawdown_pct") { max_drawdown_pct = mdd.extract().unwrap_or(0.0); }
        if let Ok(Some(fdict_any)) = p.get_item("funding") {
            if let Ok(fdict) = fdict_any.downcast::<PyDict>() {
                for (k, v) in fdict.iter() {
                    if let (Ok(pair), Ok(list_any)) = (k.extract::<String>(), v.downcast::<PyList>()) {
                        let mut vec: Vec<(i64, f64)> = Vec::new();
                        for item in list_any.iter() {
                            if let Ok(tpl) = item.downcast::<PyList>() {
                                if tpl.len() >= 2 {
                                    let ts_any = tpl.get_item(0)?;
                                    let rate_any = tpl.get_item(1)?;
                                    let ts: i64 = ts_any.extract().unwrap_or(0_i64);
                                    let rate: f64 = rate_any.extract().unwrap_or(0.0);
                                    vec.push((ts, rate));
                                }
                            }
                        }
                        if !vec.is_empty() { funding_map.insert(pair, vec); }
                    }
                }
            }
        }
        if let Ok(Some(mdict_any)) = p.get_item("funding_mark") {
            if let Ok(mdict) = mdict_any.downcast::<PyDict>() {
                for (k, v) in mdict.iter() {
                    if let (Ok(pair), Ok(list_any)) = (k.extract::<String>(), v.downcast::<PyList>()) {
                        let mut vec: Vec<(i64, f64)> = Vec::new();
                        for item in list_any.iter() {
                            if let Ok(tpl) = item.downcast::<PyList>() {
                                if tpl.len() >= 2 {
                                    let ts: i64 = tpl.get_item(0)?.extract().unwrap_or(0_i64);
                                    let mp: f64 = tpl.get_item(1)?.extract().unwrap_or(0.0);
                                    vec.push((ts, mp));
                                }
                            }
                        }
                        if !vec.is_empty() { mark_map.insert(pair, vec); }
                    }
                }
            }
        }
    }

    // Preconvert python rows to native structs to allow parallel processing
    #[derive(Clone)]
    struct Row {
        ts: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        enter_long: i32,
        exit_long: i32,
        enter_short: i32,
        exit_short: i32,
        enter_tag: Option<String>,
        exit_tag: Option<String>,
    }

    let mut data_native: Vec<(String, Vec<Row>)> = Vec::new();
    for (pair_obj, rows_obj) in data.iter() {
        let pair: String = pair_obj.extract()?;
        let rows: &PyList = rows_obj.downcast()?;
        let mut vec: Vec<Row> = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let r: &PyList = row.downcast()?;
            if r.len() < 11 { return Err(PyValueError::new_err("Row does not have 11 elements")); }
            let ts: i64 = r.get_item(0)?.extract().unwrap_or(0_i64);
            if ts < start_ts || ts > end_ts { continue; }
            let open: f64 = r.get_item(1)?.extract().unwrap_or(0.0);
            let high: f64 = r.get_item(2)?.extract().unwrap_or(open);
            let low: f64 = r.get_item(3)?.extract().unwrap_or(open);
            let close: f64 = r.get_item(4)?.extract().unwrap_or(open);
            let enter_long: i32 = r.get_item(5)?.extract().unwrap_or(0);
            let exit_long: i32 = r.get_item(6)?.extract().unwrap_or(0);
            let enter_short: i32 = r.get_item(7)?.extract().unwrap_or(0);
            let exit_short: i32 = r.get_item(8)?.extract().unwrap_or(0);
            let enter_tag: Option<String> = r.get_item(9)?.extract().ok();
            let exit_tag: Option<String> = r.get_item(10)?.extract().ok();
            vec.push(Row { ts, open, high, low, close, enter_long, exit_long, enter_short, exit_short, enter_tag, exit_tag });
        }
        data_native.push((pair, vec));
    }

    #[derive(Clone)]
    struct TradePlain {
        pair: String,
        open_ts: i64,
        close_ts: i64,
        open_rate: f64,
        close_rate: f64,
        amount: f64,
        profit_abs: f64,
        profit_ratio: f64,
        enter_tag: Option<String>,
        exit_reason: &'static str,
        fee_open: f64,
        fee_close: f64,
    }

    // Simulate per pair (in parallel)
    // TSL parameters (copy to plain values to avoid capturing Py objects in threads)
    let trailing_stop = params.and_then(|p| p.get_item("trailing_stop").ok().flatten()).map(|x| x.extract().unwrap_or(false)).unwrap_or(false);
    let trailing_stop_positive: Option<f64> = params.and_then(|p| p.get_item("trailing_stop_positive").ok().flatten()).and_then(|x| x.extract().ok());
    let trailing_stop_positive_offset: f64 = params.and_then(|p| p.get_item("trailing_stop_positive_offset").ok().flatten()).and_then(|x| x.extract().ok()).unwrap_or(0.0);
    let trailing_only_offset_is_reached: bool = params.and_then(|p| p.get_item("trailing_only_offset_is_reached").ok().flatten()).and_then(|x| x.extract().ok()).unwrap_or(false);

    // Global drawdown trackers
    let total_micro = Arc::new(AtomicI64::new(0));
    let hwm_micro = Arc::new(AtomicI64::new(0));
    let halt_flag = Arc::new(AtomicBool::new(false));

    let trades_all: Vec<TradePlain> = data_native
        .par_iter()
        .flat_map_iter(|(pair, rows)| {
            let mut res: Vec<TradePlain> = Vec::new();
            if rows.is_empty() { return res; }
            let mut in_position = false;
            let mut cooldown_until: i64 = 0;
            let mut entry_ts: i64 = 0;
            let mut entry_price: f64 = 0.0;
            let mut entry_tag: Option<String> = None;
            let mut highest: f64 = 0.0; // for TSL long
            let mut lowest: f64 = f64::MAX; // for TSL short
            let mut is_short_pos: bool = false;
            let fvec = funding_map.get(pair).cloned().unwrap_or_default();
            let mvec = mark_map.get(pair).cloned().unwrap_or_default();
            let mut fidx: usize = 0;
            let mut midx: usize = 0;
            let mut funding_acc: f64 = 0.0;
            let mut sl_events: Vec<i64> = Vec::new();

            for r in rows {
                if !in_position {
                    if max_drawdown_pct > 0.0 && halt_flag.load(Ordering::SeqCst) { continue; }
                    if cooldown_minutes > 0 && r.ts < cooldown_until { continue; }
                    if r.enter_long != 0 {
                        in_position = true;
                        is_short_pos = false;
                        entry_ts = r.ts;
                        let mut ep = r.open * (1.0 + slippage);
                        if ep > r.high { ep = r.high; }
                        entry_price = ep;
                        entry_tag = r.enter_tag.clone();
                        highest = entry_price.max(r.high);
                        lowest = r.low.min(entry_price);
                        continue;
                    } else if can_short && r.enter_short != 0 {
                        in_position = true;
                        is_short_pos = true;
                        entry_ts = r.ts;
                        let mut ep = r.open * (1.0 - slippage);
                        if ep < r.low { ep = r.low; }
                        entry_price = ep;
                        entry_tag = r.enter_tag.clone();
                        highest = r.high.max(entry_price);
                        lowest = entry_price.min(r.low);
                        continue;
                    }
                }
                if in_position {
                    let mut exit_rate_opt: Option<f64> = None;
                    let mut reason: &'static str = "exit_signal";
                    // Accrue funding at this timestamp if present
                    if fidx < fvec.len() {
                        while fidx < fvec.len() && fvec[fidx].0 < r.ts { fidx += 1; }
                        if fidx < fvec.len() && fvec[fidx].0 == r.ts {
                            let rate = fvec[fidx].1;
                            // find mark at same ts if available
                            let mut notional = r.open;
                            if !mvec.is_empty() {
                                while midx < mvec.len() && mvec[midx].0 < r.ts { midx += 1; }
                                if midx < mvec.len() && mvec[midx].0 == r.ts {
                                    notional = mvec[midx].1;
                                }
                            }
                            let ff = notional * rate; // amount=1.0
                            // long pays positive rate; short receives
                            if !is_short_pos { funding_acc += ff; } else { funding_acc -= ff; }
                            fidx += 1;
                        }
                    }
                    if !is_short_pos {
                        if r.high > highest { highest = r.high; }
                        if r.low < lowest { lowest = r.low; }
                        let stop_price = entry_price * (1.0 + stoploss);
                        if r.low <= stop_price { exit_rate_opt = Some(stop_price); reason = "stoploss"; }
                        else {
                            let elapsed_min = ((r.ts - entry_ts) / 60).max(0);
                            if let Some(roi_req) = roi.target_for_minutes(elapsed_min) {
                                let roi_price = entry_price * (1.0 + roi_req);
                                if r.high >= roi_price { exit_rate_opt = Some(roi_price); reason = "roi"; }
                            }
                            if exit_rate_opt.is_none() && trailing_stop {
                                if let Some(ts_pos) = trailing_stop_positive {
                                    let reached_offset = ((highest / entry_price) - 1.0) >= trailing_stop_positive_offset;
                                    if (!trailing_only_offset_is_reached) || reached_offset {
                                        let tsl_price = highest * (1.0 - ts_pos);
                                        if r.low <= tsl_price { exit_rate_opt = Some(tsl_price); reason = "trailing_stop"; }
                                    }
                                }
                            }
                        }
                        if exit_rate_opt.is_none() && r.exit_long != 0 {
                            let mut er = r.open; if er < r.low { er = r.low; }
                            exit_rate_opt = Some(er); reason = "exit_signal";
                        }
                    } else {
                        if r.high > highest { highest = r.high; }
                        if r.low < lowest { lowest = r.low; }
                        let stop_price = entry_price * (1.0 - stoploss);
                        if r.high >= stop_price { exit_rate_opt = Some(stop_price); reason = "stoploss"; }
                        else {
                            let elapsed_min = ((r.ts - entry_ts) / 60).max(0);
                            if let Some(roi_req) = roi.target_for_minutes(elapsed_min) {
                                let roi_price = entry_price * (1.0 - roi_req);
                                if r.low <= roi_price { exit_rate_opt = Some(roi_price); reason = "roi"; }
                            }
                            if exit_rate_opt.is_none() && trailing_stop {
                                if let Some(ts_pos) = trailing_stop_positive {
                                    let reached_offset = ((entry_price / lowest) - 1.0) >= trailing_stop_positive_offset;
                                    if (!trailing_only_offset_is_reached) || reached_offset {
                                        let tsl_price = lowest * (1.0 + ts_pos);
                                        if r.high >= tsl_price { exit_rate_opt = Some(tsl_price); reason = "trailing_stop"; }
                                    }
                                }
                            }
                        }
                        if exit_rate_opt.is_none() && can_short && r.exit_short != 0 {
                            let mut er = r.open; if er > r.high { er = r.high; }
                            exit_rate_opt = Some(er); reason = "exit_signal";
                        }
                    }

                    if let Some(mut exit_rate) = exit_rate_opt {
                        let amount = 1.0_f64;
                        let (fee_open_abs, fee_close_abs, profit_abs, profit_ratio, final_exit_rate) = if !is_short_pos {
                            let ex = exit_rate * (1.0 - slippage);
                            let fee_open_abs = (entry_price * amount) * fee;
                            let fee_close_abs = (ex * amount) * fee;
                            let gross = (ex - entry_price) * amount;
                            let profit_abs = gross - (fee_open_abs + fee_close_abs) - funding_acc;
                            let profit_ratio = if entry_price != 0.0 { profit_abs / (entry_price * amount) } else { 0.0 };
                            (fee_open_abs, fee_close_abs, profit_abs, profit_ratio, ex)
                        } else {
                            let ex = exit_rate * (1.0 + slippage);
                            let fee_open_abs = (entry_price * amount) * fee;
                            let fee_close_abs = (ex * amount) * fee;
                            let gross = (entry_price - ex) * amount;
                            let profit_abs = gross - (fee_open_abs + fee_close_abs) - funding_acc;
                            let profit_ratio = if entry_price != 0.0 { profit_abs / (entry_price * amount) } else { 0.0 };
                            (fee_open_abs, fee_close_abs, profit_abs, profit_ratio, ex)
                        };
                        res.push(TradePlain {
                            pair: pair.clone(),
                            open_ts: entry_ts,
                            close_ts: r.ts,
                            open_rate: entry_price,
                            close_rate: final_exit_rate,
                            amount,
                            profit_abs,
                            profit_ratio,
                            enter_tag: entry_tag.clone(),
                            exit_reason: reason,
                            fee_open: fee_open_abs,
                            fee_close: fee_close_abs,
                        });
                        in_position = false; entry_ts = 0; entry_price = 0.0; entry_tag = None; highest = 0.0; lowest = f64::MAX; is_short_pos = false; funding_acc = 0.0;
                        if reason == "stoploss" { sl_events.push(r.ts); }
                        if cooldown_minutes > 0 { cooldown_until = r.ts + cooldown_minutes * 60; }
                        if max_drawdown_pct > 0.0 {
                            let pmicro = (profit_abs * 1_000_000.0).round() as i64;
                            let total_now = total_micro.fetch_add(pmicro, Ordering::SeqCst) + pmicro;
                            // update HWM
                            loop {
                                let cur = hwm_micro.load(Ordering::SeqCst);
                                if total_now > cur {
                                    if hwm_micro.compare_exchange(cur, total_now, Ordering::SeqCst, Ordering::SeqCst).is_ok() { break; } else { continue; }
                                }
                                break;
                            }
                            let hwm_now = hwm_micro.load(Ordering::SeqCst);
                            if hwm_now > 0 {
                                let drop = (hwm_now - total_now) as f64 / (hwm_now as f64);
                                if drop >= max_drawdown_pct { halt_flag.store(true, Ordering::SeqCst); }
                            }
                        }
                    }
                }
            }
            res
        })
        .collect();

    // Convert to Python list
    for t in trades_all {
        let td = trade_dict(
            py,
            &t.pair,
            t.open_ts,
            t.close_ts,
            t.open_rate,
            t.close_rate,
            t.amount,
            t.profit_abs,
            t.profit_ratio,
            t.enter_tag.as_deref(),
            t.fee_open,
            t.fee_close,
        )?;
        td.set_item("exit_reason", t.exit_reason)?;
        out.append(td)?;
    }

    Ok(out)
}

#[pymodule]
fn ft_rust_backtest(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trades, m)?)?;
    Ok(())
}
