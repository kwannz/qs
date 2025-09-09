# Freqtrade Numba Optimization - Deployment & Configuration Guide

## Overview

This guide provides comprehensive instructions for deploying and configuring the Numba optimization system in Freqtrade environments, from development to production.

## 1. System Requirements

### 1.1 Hardware Requirements

**Minimum Requirements:**
- CPU: 2+ cores with x86_64 architecture
- RAM: 4GB available memory  
- Storage: 1GB free space for numba cache

**Recommended for Production:**
- CPU: 4+ cores with AVX2 support
- RAM: 8GB+ available memory
- Storage: 2GB+ SSD space for optimal cache performance
- Network: Stable connection for real-time trading

### 1.2 Software Dependencies

**Core Dependencies:**
```
numba>=0.57.0
numpy>=1.21.0
pandas>=1.5.0
llvmlite>=0.40.0
```

**Optional Performance Dependencies:**
```
mkl>=2022.0.0              # Intel Math Kernel Library
mkl-service>=2.4.0         # MKL threading control
tbb>=2021.5.0              # Intel Threading Building Blocks
```

### 1.3 Python Environment Setup

```bash
# Create isolated environment
python -m venv venv_numba
source venv_numba/bin/activate  # Linux/Mac
# or venv_numba\Scripts\activate  # Windows

# Install core dependencies
pip install numba numpy pandas

# Verify installation
python -c "import numba; print(f'Numba {numba.__version__} installed successfully')"
```

## 2. Configuration Management

### 2.1 Configuration Schema

Add the following configuration options to your Freqtrade config:

```json
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "numba_cache_directory": "user_data/numba_cache",
    "numba_parallel_threads": "auto",
    "performance_benchmarking": false,
    "fallback_on_numba_error": true,
    "optimization_phase": "progressive",
    "enabled_modules": ["metrics", "indicators"],
    "detailed_performance_logging": false,
    "cache_persistence": true,
    "compilation_timeout": 300,
    "memory_limit_mb": 2048
  }
}
```

### 2.2 Configuration Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_numba_optimization` | bool | `true` | Master switch for all numba optimizations |
| `numba_cache_directory` | string | `"user_data/numba_cache"` | Directory for compiled function cache |
| `numba_parallel_threads` | string/int | `"auto"` | Number of threads for parallel execution |
| `performance_benchmarking` | bool | `false` | Enable detailed performance metrics collection |
| `fallback_on_numba_error` | bool | `true` | Automatically fallback on optimization errors |
| `optimization_phase` | string | `"conservative"` | Deployment phase: conservative/progressive/aggressive |
| `enabled_modules` | array | `["metrics"]` | List of modules with optimization enabled |
| `detailed_performance_logging` | bool | `false` | Enable verbose performance logging |
| `cache_persistence` | bool | `true` | Persist compiled functions across sessions |
| `compilation_timeout` | int | `300` | Timeout for function compilation (seconds) |
| `memory_limit_mb` | int | `2048` | Memory limit for optimization processes |

### 2.3 Environment Variables

Set these environment variables to control numba behavior:

```bash
# Production Environment
export NUMBA_CACHE_DIR="$HOME/.freqtrade/numba_cache"
export NUMBA_NUM_THREADS="auto"
export NUMBA_THREADING_LAYER="tbb"  # or "omp", "workqueue"
export NUMBA_DISABLE_JIT="0"        # Ensure JIT is enabled
export NUMBA_WARNINGS="0"           # Suppress non-critical warnings

# Development Environment  
export NUMBA_CACHE_DIR="./dev_cache"
export NUMBA_DEBUG="1"               # Enable debug info
export NUMBA_DEBUG_FRONTEND="1"      # Enable frontend debug
export NUMBA_DISABLE_PERFORMANCE_WARNINGS="1"
```

## 3. Deployment Strategies

### 3.1 Conservative Deployment (Phase 1)

**Target**: Low-risk production environments
**Timeline**: Week 1-2 
**Features**: Core metrics optimizations only

```json
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "optimization_phase": "conservative", 
    "enabled_modules": ["metrics"],
    "performance_benchmarking": false,
    "fallback_on_numba_error": true,
    "detailed_performance_logging": false
  }
}
```

**Validation Checklist:**
- [ ] Basic metrics functions (market_change, combine_dataframes) working
- [ ] No performance regression in backtesting
- [ ] Fallback mechanisms tested
- [ ] Memory usage within acceptable limits
- [ ] Cache directory permissions correct

### 3.2 Progressive Deployment (Phase 2)  

**Target**: Environments ready for moderate optimization
**Timeline**: Week 3-4
**Features**: Metrics + basic indicators

```json
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "optimization_phase": "progressive",
    "enabled_modules": ["metrics", "indicators"],
    "performance_benchmarking": true,
    "fallback_on_numba_error": true,
    "detailed_performance_logging": true
  }
}
```

**Validation Checklist:**
- [ ] All Phase 1 validations passed
- [ ] Technical indicators (SMA, RSI, Bollinger Bands) optimized
- [ ] Strategy performance unchanged
- [ ] Performance improvements documented
- [ ] Error rates < 0.1%

### 3.3 Aggressive Deployment (Phase 3)

**Target**: High-performance environments with full optimization
**Timeline**: Week 5-6
**Features**: Full optimization suite

```json
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "optimization_phase": "aggressive",
    "enabled_modules": ["metrics", "indicators", "backtesting"],
    "performance_benchmarking": true,
    "fallback_on_numba_error": true,
    "detailed_performance_logging": false,
    "numba_parallel_threads": 4
  }
}
```

**Validation Checklist:**
- [ ] All previous phases validated
- [ ] Backtesting acceleration enabled
- [ ] Parallel processing working correctly
- [ ] No memory leaks or stability issues
- [ ] Performance gains measured and documented

## 4. Environment-Specific Configurations

### 4.1 Development Environment

```bash
#!/bin/bash
# development_setup.sh

# Set development environment variables
export FREQTRADE_ENV="development"
export NUMBA_CACHE_DIR="./dev_numba_cache" 
export NUMBA_DEBUG="1"
export NUMBA_DEBUG_FRONTEND="1"
export NUMBA_DISABLE_PERFORMANCE_WARNINGS="1"

# Enable development configuration
cat > dev_config_overlay.json << EOF
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "optimization_phase": "experimental", 
    "performance_benchmarking": true,
    "detailed_performance_logging": true,
    "cache_persistence": false,
    "compilation_timeout": 600
  }
}
EOF

echo "Development environment configured"
echo "Cache directory: $NUMBA_CACHE_DIR"
echo "Debug mode: enabled"
```

### 4.2 Testing Environment

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  freqtrade-test:
    image: freqtradeorg/freqtrade:latest
    environment:
      - NUMBA_CACHE_DIR=/freqtrade/user_data/numba_cache
      - NUMBA_NUM_THREADS=2
      - NUMBA_THREADING_LAYER=workqueue
      - NUMBA_DISABLE_JIT=0
    volumes:
      - ./test_config.json:/freqtrade/config.json
      - ./user_data:/freqtrade/user_data
    command: >
      freqtrade backtesting
      --config config.json
      --strategy-path user_data/strategies
      --datadir user_data/data
      --timeframe 5m
      --timerange 20230101-20231201
```

### 4.3 Production Environment

```bash
#!/bin/bash
# production_setup.sh

# Production environment variables
export FREQTRADE_ENV="production"
export NUMBA_CACHE_DIR="/opt/freqtrade/cache/numba"
export NUMBA_NUM_THREADS="auto"
export NUMBA_THREADING_LAYER="tbb"
export NUMBA_DISABLE_JIT="0"
export NUMBA_WARNINGS="0"

# Create cache directory with proper permissions
sudo mkdir -p $NUMBA_CACHE_DIR
sudo chown freqtrade:freqtrade $NUMBA_CACHE_DIR
sudo chmod 755 $NUMBA_CACHE_DIR

# Set resource limits
ulimit -v 4194304  # 4GB virtual memory limit
ulimit -m 2097152  # 2GB physical memory limit

# Production configuration
cat > prod_config_overlay.json << EOF
{
  "performance_optimization": {
    "enable_numba_optimization": true,
    "optimization_phase": "aggressive",
    "performance_benchmarking": false,
    "detailed_performance_logging": false,
    "fallback_on_numba_error": true,
    "memory_limit_mb": 2048,
    "compilation_timeout": 300
  }
}
EOF

echo "Production environment configured"
echo "Cache directory: $NUMBA_CACHE_DIR"
echo "Threading layer: TBB"
echo "Memory limits applied"
```

## 5. Performance Monitoring Setup

### 5.1 Metrics Collection Configuration

```python
# user_data/performance_monitoring.py
import logging
from datetime import datetime
from pathlib import Path

class PerformanceMonitor:
    def __init__(self, config_path: str = "user_data/performance_metrics.json"):
        self.config_path = Path(config_path)
        self.metrics_log = []
        
    def setup_monitoring(self):
        """Setup performance monitoring with appropriate log levels"""
        
        # Configure logging
        logging.getLogger('freqtrade.optimize.numba_manager').setLevel(logging.INFO)
        logging.getLogger('freqtrade.optimize.performance_proxy').setLevel(logging.WARNING)
        
        # Setup periodic reporting
        from apscheduler import schedulers
        scheduler = schedulers.blocking.BlockingScheduler()
        
        @scheduler.scheduled_job('interval', minutes=30)
        def log_performance_metrics():
            # Collect and log performance metrics
            self.collect_metrics()
        
        scheduler.start()
    
    def collect_metrics(self):
        """Collect current performance metrics"""
        from freqtrade.optimize.numba_manager import get_numba_manager
        
        manager = get_numba_manager()
        if manager:
            report = manager.get_performance_report()
            
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'performance_report': report,
                'system_info': self.get_system_info()
            }
            
            self.metrics_log.append(metric_entry)
            self.save_metrics()
    
    def get_system_info(self):
        """Get relevant system information"""
        import psutil
        import os
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'cache_size_mb': self.get_cache_size(),
            'numba_threads': os.environ.get('NUMBA_NUM_THREADS', 'auto')
        }
    
    def get_cache_size(self):
        """Get numba cache directory size in MB"""
        cache_dir = Path(os.environ.get('NUMBA_CACHE_DIR', './numba_cache'))
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)  # Convert to MB
        return 0
```

### 5.2 Alerting Configuration

```python
# user_data/performance_alerts.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class PerformanceAlerts:
    def __init__(self, config: dict):
        self.config = config
        self.alert_thresholds = {
            'fallback_rate_threshold': 0.05,  # 5% fallback rate triggers alert
            'performance_regression_threshold': 0.2,  # 20% performance drop
            'memory_usage_threshold': 0.8,  # 80% memory usage
            'compilation_failure_threshold': 0.1  # 10% compilation failures
        }
    
    def check_performance_alerts(self, performance_report: dict):
        """Check performance metrics against alert thresholds"""
        
        alerts = []
        
        for func_name, stats in performance_report.items():
            # Check fallback rate
            fallback_rate = stats.get('fallback_rate', 0)
            if fallback_rate > self.alert_thresholds['fallback_rate_threshold']:
                alerts.append({
                    'type': 'HIGH_FALLBACK_RATE',
                    'function': func_name,
                    'current_rate': fallback_rate,
                    'threshold': self.alert_thresholds['fallback_rate_threshold'],
                    'severity': 'WARNING'
                })
            
            # Check performance regression  
            if 'speedup' in stats and stats['speedup'] < 1.0:
                regression = 1.0 - stats['speedup']
                if regression > self.alert_thresholds['performance_regression_threshold']:
                    alerts.append({
                        'type': 'PERFORMANCE_REGRESSION',
                        'function': func_name,
                        'current_speedup': stats['speedup'],
                        'regression_pct': regression * 100,
                        'severity': 'CRITICAL'
                    })
        
        if alerts:
            self.send_alerts(alerts)
    
    def send_alerts(self, alerts: list):
        """Send performance alerts via configured channels"""
        
        for alert in alerts:
            if alert['severity'] == 'CRITICAL':
                self.send_email_alert(alert)
                self.log_alert(alert)
            else:
                self.log_alert(alert)
    
    def send_email_alert(self, alert: dict):
        """Send email alert for critical issues"""
        
        if not self.config.get('email_alerts_enabled', False):
            return
        
        smtp_config = self.config.get('smtp', {})
        if not smtp_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email'] 
            msg['Subject'] = f"Freqtrade Numba Alert: {alert['type']}"
            
            body = f"""
            Performance Alert: {alert['type']}
            
            Function: {alert.get('function', 'N/A')}
            Severity: {alert['severity']}
            Details: {alert}
            
            Time: {datetime.now()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def log_alert(self, alert: dict):
        """Log alert to system logs"""
        severity_map = {
            'INFO': logging.info,
            'WARNING': logging.warning, 
            'CRITICAL': logging.critical
        }
        
        log_func = severity_map.get(alert['severity'], logging.warning)
        log_func(f"Performance Alert: {alert}")
```

## 6. Troubleshooting and Maintenance

### 6.1 Common Issues and Solutions

**Issue: Numba compilation failures**
```bash
# Solution: Clear cache and recompile
rm -rf $NUMBA_CACHE_DIR/*
export NUMBA_DEBUG=1
python -c "from freqtrade.optimize.numba_manager import get_numba_manager; get_numba_manager()"
```

**Issue: High memory usage**
```bash
# Solution: Limit parallel threads and cache size
export NUMBA_NUM_THREADS=2
export NUMBA_CACHE_DIR="/tmp/numba_cache_small"
ulimit -v 2097152  # 2GB limit
```

**Issue: Performance regression**
```python
# Solution: Run performance regression analysis
from freqtrade.optimize.testing_framework import RegressionTestSuite

suite = RegressionTestSuite('user_data/performance_baseline.json')
current_results = run_performance_benchmarks()
regression_report = suite.check_regression(current_results)
print(regression_report)
```

### 6.2 Cache Management

```bash
#!/bin/bash
# cache_maintenance.sh

CACHE_DIR=${NUMBA_CACHE_DIR:-"user_data/numba_cache"}
MAX_CACHE_SIZE_MB=1024
MAX_CACHE_AGE_DAYS=7

# Check cache size
current_size=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)

if [ "$current_size" -gt "$MAX_CACHE_SIZE_MB" ]; then
    echo "Cache size ($current_size MB) exceeds limit ($MAX_CACHE_SIZE_MB MB)"
    echo "Cleaning old cache files..."
    
    # Remove files older than MAX_CACHE_AGE_DAYS
    find "$CACHE_DIR" -type f -mtime +$MAX_CACHE_AGE_DAYS -delete
    
    # If still too large, remove oldest files
    if [ "$(du -sm "$CACHE_DIR" | cut -f1)" -gt "$MAX_CACHE_SIZE_MB" ]; then
        echo "Removing oldest cache files..."
        find "$CACHE_DIR" -type f -printf '%T@ %p\n' | sort -n | head -50 | cut -d' ' -f2- | xargs rm -f
    fi
fi

echo "Cache maintenance completed"
echo "Current cache size: $(du -sh "$CACHE_DIR" | cut -f1)"
```

### 6.3 Performance Validation Scripts

```python
#!/usr/bin/env python3
# validate_performance.py

import sys
import time
from pathlib import Path

def validate_optimization_setup():
    """Validate that optimization setup is working correctly"""
    
    print("Freqtrade Numba Optimization - Validation Suite")
    print("=" * 60)
    
    # 1. Check numba availability
    print("1. Checking numba availability...")
    try:
        import numba
        print(f"   ✓ Numba {numba.__version__} is available")
    except ImportError as e:
        print(f"   ✗ Numba not available: {e}")
        return False
    
    # 2. Test basic numba functionality
    print("2. Testing basic numba functionality...")
    try:
        from numba import njit
        
        @njit
        def test_function(x):
            return x * 2
        
        result = test_function(21)
        if result == 42:
            print("   ✓ Basic numba compilation working")
        else:
            print(f"   ✗ Basic numba test failed: expected 42, got {result}")
            return False
    except Exception as e:
        print(f"   ✗ Basic numba test failed: {e}")
        return False
    
    # 3. Check cache directory
    print("3. Checking cache directory...")
    cache_dir = Path(os.environ.get('NUMBA_CACHE_DIR', 'user_data/numba_cache'))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_file = cache_dir / '.test'
        test_file.write_text('test')
        test_file.unlink()
        print(f"   ✓ Cache directory accessible: {cache_dir}")
    except Exception as e:
        print(f"   ✗ Cache directory not accessible: {e}")
        return False
    
    # 4. Test numba manager initialization
    print("4. Testing NumbaManager initialization...")
    try:
        from implementation_examples import NumbaManager
        
        config = {
            'enable_numba_optimization': True,
            'numba_optimization_modules': ['metrics'],
            'performance_benchmarking': False
        }
        
        manager = NumbaManager(config)
        if manager.numba_available:
            print("   ✓ NumbaManager initialized successfully")
        else:
            print("   ⚠ NumbaManager initialized but numba not available")
            return False
    except Exception as e:
        print(f"   ✗ NumbaManager initialization failed: {e}")
        return False
    
    # 5. Test performance improvement
    print("5. Testing performance improvement...")
    try:
        from testing_framework_example import MetricsBenchmark
        
        benchmark = MetricsBenchmark([1000])
        results = benchmark.run_full_benchmark()
        
        if 1000 in results and 'speedup' in results[1000]:
            speedup = results[1000]['speedup']
            if speedup > 1.0:
                print(f"   ✓ Performance improvement detected: {speedup:.2f}x speedup")
            else:
                print(f"   ⚠ No performance improvement: {speedup:.2f}x speedup")
        else:
            print("   ⚠ Could not measure performance improvement")
    except Exception as e:
        print(f"   ✗ Performance test failed: {e}")
    
    print("\nValidation completed successfully! ✓")
    return True

if __name__ == '__main__':
    import os
    
    # Set up environment if needed
    if 'NUMBA_CACHE_DIR' not in os.environ:
        os.environ['NUMBA_CACHE_DIR'] = './validation_cache'
    
    success = validate_optimization_setup()
    sys.exit(0 if success else 1)
```

## 7. Deployment Checklist

### 7.1 Pre-Deployment Checklist

- [ ] **Environment Validation**
  - [ ] Python environment with all dependencies installed
  - [ ] Numba version compatibility verified (>=0.57.0)
  - [ ] System resources meet minimum requirements
  - [ ] Cache directory permissions configured correctly

- [ ] **Configuration Validation**
  - [ ] Performance optimization config section added
  - [ ] Environment variables set appropriately
  - [ ] Phase-appropriate settings selected
  - [ ] Fallback mechanisms configured

- [ ] **Testing Validation**
  - [ ] All correctness tests passing
  - [ ] Performance benchmarks show improvement
  - [ ] Integration tests completed successfully
  - [ ] Memory usage within acceptable limits

- [ ] **Monitoring Setup**
  - [ ] Performance monitoring configured
  - [ ] Alert thresholds set appropriately
  - [ ] Log rotation configured
  - [ ] Metrics collection enabled (if desired)

### 7.2 Post-Deployment Checklist

- [ ] **Operational Validation**
  - [ ] Trading strategies producing expected results
  - [ ] No performance regressions detected
  - [ ] Error rates within acceptable limits (<0.1%)
  - [ ] Memory usage stable

- [ ] **Monitoring Verification**
  - [ ] Performance metrics being collected
  - [ ] Alert system tested and working
  - [ ] Log files being generated appropriately
  - [ ] Cache maintenance running correctly

- [ ] **Documentation Updates**
  - [ ] Deployment notes recorded
  - [ ] Configuration changes documented
  - [ ] Performance improvements quantified
  - [ ] Known issues and workarounds documented

## 8. Rollback Procedures

### 8.1 Immediate Rollback (Emergency)

```bash
#!/bin/bash
# emergency_rollback.sh

echo "Executing emergency rollback of numba optimizations..."

# Disable optimizations immediately
export NUMBA_DISABLE_JIT=1

# Update configuration
sed -i 's/"enable_numba_optimization": true/"enable_numba_optimization": false/' config.json

# Restart freqtrade process
pkill -f freqtrade
sleep 5
nohup freqtrade trade --config config.json &

echo "Emergency rollback completed"
echo "Numba optimizations disabled"
echo "Freqtrade restarted with fallback implementations"
```

### 8.2 Planned Rollback

```bash
#!/bin/bash
# planned_rollback.sh

echo "Executing planned rollback of numba optimizations..."

# 1. Stop trading (gracefully)
freqtrade stop

# 2. Backup current configuration
cp config.json config.json.backup.$(date +%Y%m%d_%H%M%S)

# 3. Update configuration to disable optimizations
python << EOF
import json
with open('config.json', 'r') as f:
    config = json.load(f)

config['performance_optimization']['enable_numba_optimization'] = False
config['performance_optimization']['optimization_phase'] = 'disabled'

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF

# 4. Clear numba cache
rm -rf $NUMBA_CACHE_DIR/*

# 5. Restart with original implementations
freqtrade trade --config config.json

echo "Planned rollback completed successfully"
```

This comprehensive deployment and configuration guide provides everything needed to successfully deploy the Numba optimization system in Freqtrade environments, from initial setup through production deployment and maintenance.