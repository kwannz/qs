use anyhow::Result;
use async_nats::Client;
use serde::Serialize;
use std::{collections::HashMap, path::PathBuf};
use tokio::{fs, io::{AsyncBufReadExt, BufReader}, time::{sleep, Duration}};
use tracing::{info, warn};

#[derive(Serialize)]
struct Tick<'a> {
    symbol: &'a str,
    price: f64,
    ts: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://nats:4222".to_string());
    let sleep_secs: u64 = std::env::var("SLEEP_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    info!("Starting data-bridge: data_dir={}, nats_url={}", data_dir, nats_url);
    let client = async_nats::connect(nats_url).await?;
    let mut offsets: HashMap<PathBuf, usize> = HashMap::new();

    loop {
        ingest_dir(&client, &data_dir, &mut offsets).await?;
        sleep(Duration::from_secs(sleep_secs)).await;
    }
}

async fn ingest_dir(client: &Client, dir: &str, offsets: &mut HashMap<PathBuf, usize>) -> Result<()> {
    let mut rd = fs::read_dir(dir).await?;
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if entry.file_type().await?.is_dir() { continue; }
        let name = path.to_string_lossy().to_string();
        if !(name.ends_with(".csv") || name.ends_with(".json") || name.ends_with(".ndjson")) {
            continue;
        }
        let file = fs::File::open(&path).await?;
        let mut reader = BufReader::new(file).lines();
        let mut idx: usize = 0;
        let start = *offsets.get(&path).unwrap_or(&0);
        let mut processed = 0usize;
        while let Some(line) = reader.next_line().await? {
            if idx < start { idx += 1; continue; }
            if let Some((sym, price, ts)) = parse_line(&line) {
                let tick = Tick { symbol: &sym, price, ts };
                if let Ok(bytes) = serde_json::to_vec(&tick) {
                    if let Err(e) = client.publish("market.ticks", bytes.into()).await { warn!("nats publish error: {}", e); }
                }
            }
            idx += 1; processed += 1;
        }
        offsets.insert(path.clone(), idx);
        if processed > 0 { info!("ingested {} new lines from {}", processed, name); }
    }
    Ok(())
}

fn parse_line(line: &str) -> Option<(String, f64, Option<String>)> {
    let s = line.trim();
    if s.is_empty() { return None; }
    if s.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let symbol = v.get("symbol")?.as_str()?.to_string();
        let price = v.get("price")?.as_f64()?;
        let ts = v.get("ts").and_then(|t| t.as_str().map(|x| x.to_string()));
        Some((symbol, price, ts))
    } else {
        // csv ts,symbol,price
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() < 3 { return None; }
        let ts = Some(parts[0].trim().to_string());
        let symbol = parts[1].trim().to_string();
        let price = parts[2].trim().parse::<f64>().ok()?;
        Some((symbol, price, ts))
    }
}
