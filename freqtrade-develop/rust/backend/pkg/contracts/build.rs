use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Use vendored protoc to avoid system dependency
    let protoc_path = protoc_bin_vendored::protoc_bin_path().expect("Failed to fetch vendored protoc");
    env::set_var("PROTOC", protoc_path);
    
    // Sprint 1: Configure tonic-build for v1 service definitions
    tonic_build::configure()
        .build_server(cfg!(feature = "server"))
        .build_client(cfg!(feature = "client"))
        .out_dir(&out_dir)
        // Remove blanket serde derives to avoid conflicts with well-known types
        .compile_protos(
            &[
                // Sprint 1 v1 service definitions
                "proto/trading/v1/common.proto",
                "proto/trading/v1/execution.proto",
                "proto/trading/v1/backtest.proto", 
                "proto/trading/v1/factor.proto",
                "proto/trading/v1/risk.proto",
                "proto/trading/v1/markets.proto",
                // Event system protocols
                "proto/events/v1/events.proto",
                // Legacy support
                "proto/trading.proto",
            ],
            &["proto/"],
        )?;

    // Generate type definitions for generated code
    println!("cargo:rerun-if-changed=proto/");
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
