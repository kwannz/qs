.PHONY: hdb-prep hdb-backtest hdb-fill hdb-create-profile

# Optional local overrides (not committed): define HDB_* vars here
-include Makefile.local

# Defaults (override on command line, e.g. `make hdb-prep HDB_PAIRS="BTC/USDT"`)
HDB_CONFIG ?= user_data/config.json
HDB_DATADIR ?= /Users/zhaoleon/Downloads/freqtrade-develop/historical\ data
HDB_PAIRS ?= BTC/USDT ETH/USDT
HDB_TFS ?= 5m 1h
HDB_TIMERANGE ?= 20240101-20240701
HDB_JOBS ?= 4
HDB_STRATEGY ?= SampleStrategy
HDB_REPORT_OUT ?=
HDB_REPORT_FORMAT ?= json
HDB_REPORT_DIR ?=
HDB_REPORT_BASE ?= report

PYTHON ?= python

hdb-prep:
	@echo "[HDB] Prep (dedupe + gap scan) on $(HDB_DATADIR)";
	$(PYTHON) scripts/hdb_backtest.py \
		--config "$(HDB_CONFIG)" \
		--datadir "$(HDB_DATADIR)" \
		--pairs $(HDB_PAIRS) \
		--timeframes $(HDB_TFS) \
		--timerange $(HDB_TIMERANGE) \
		--fix --jobs $(HDB_JOBS)

hdb-backtest:
	@echo "[HDB] Prep + Backtest on $(HDB_DATADIR)";
	$(PYTHON) scripts/hdb_backtest.py \
		--config "$(HDB_CONFIG)" \
		--datadir "$(HDB_DATADIR)" \
		--pairs $(HDB_PAIRS) \
		--timeframes $(HDB_TFS) \
		--timerange $(HDB_TIMERANGE) \
		--fix --jobs $(HDB_JOBS) \
		--backtest --strategy $(HDB_STRATEGY) \
		$(if $(HDB_REPORT_OUT),--report-out "$(HDB_REPORT_OUT)",) \
		$(if $(HDB_REPORT_OUT),--report-format $(HDB_REPORT_FORMAT),) \
		$(if $(HDB_REPORT_DIR),--report-dir "$(HDB_REPORT_DIR)",) \
		$(if $(HDB_REPORT_DIR),--report-base "$(HDB_REPORT_BASE)",)

hdb-fill:
	@echo "[HDB] Scan + Fill gaps on $(HDB_DATADIR)";
	$(PYTHON) scripts/fill_hdb_gaps.py \
		--config "$(HDB_CONFIG)" \
		--datadir "$(HDB_DATADIR)" \
		--pairs $(HDB_PAIRS) \
		--timeframes $(HDB_TFS) \
		--timerange $(HDB_TIMERANGE) \
		--jobs $(HDB_JOBS)

hdb-create-profile:
	@echo "Creating Makefile.local with HDB_* defaults (if not exists)";
	@if [ -f Makefile.local ]; then echo "Makefile.local already exists"; else cp Makefile.hdb.example Makefile.local; fi
