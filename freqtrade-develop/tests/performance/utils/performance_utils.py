"""
Performance testing utilities and test data generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from pathlib import Path


def generate_trading_data(years: int = 1, 
                         pairs: int = 10, 
                         frequency: str = '1h',
                         seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Generate realistic synthetic trading data for performance testing
    
    Args:
        years: Number of years of data to generate
        pairs: Number of trading pairs
        frequency: Data frequency (e.g., '1h', '5m')
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping pair names to OHLCV DataFrames
    """
    np.random.seed(seed)
    
    # Generate time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    data = {}
    
    for i in range(pairs):
        pair_name = f"PAIR{i+1}/USDT"
        
        # Generate realistic price movements
        n_periods = len(date_range)
        base_price = 100 + np.random.exponential(50)
        
        # Random walk with trend and volatility
        returns = np.random.normal(0.0002, 0.02, n_periods)  # Small positive drift
        returns = np.cumsum(returns)
        
        # Generate OHLCV data
        close_prices = base_price * np.exp(returns)
        
        # Add some intraday volatility
        volatility = np.random.uniform(0.005, 0.03, n_periods)
        high_prices = close_prices * (1 + volatility)
        low_prices = close_prices * (1 - volatility)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Generate volumes (correlated with price movements)
        volume_base = np.random.exponential(1000000, n_periods)
        volume_multiplier = 1 + np.abs(returns) * 5  # Higher volume on big moves
        volumes = volume_base * volume_multiplier
        
        df = pd.DataFrame({
            'date': date_range,
            'open': open_prices,
            'high': high_prices, 
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        data[pair_name] = df
        
    return data


def generate_trades_data(n_trades: int = 1000, 
                        timeframe: str = '1h',
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic trades data for parallelism analysis testing
    
    Args:
        n_trades: Number of trades to generate
        timeframe: Timeframe for the trades
        seed: Random seed
        
    Returns:
        DataFrame with trade data
    """
    np.random.seed(seed)
    
    # Generate trade dates over a year
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    # Generate random trade start times
    time_range_seconds = int((end_date - start_date).total_seconds())
    random_seconds = np.random.randint(0, time_range_seconds, n_trades)
    open_dates = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
    open_dates.sort()
    
    # Generate trade durations (in hours)
    durations_hours = np.random.exponential(24, n_trades)  # Avg 24 hours
    durations_hours = np.clip(durations_hours, 0.5, 240)  # 0.5h to 10 days
    
    close_dates = [open_date + timedelta(hours=duration) 
                  for open_date, duration in zip(open_dates, durations_hours)]
    
    # Generate pairs
    pair_names = [f"PAIR{np.random.randint(1, 21)}/USDT" for _ in range(n_trades)]
    
    # Generate profit/loss percentages
    profit_pct = np.random.normal(0.5, 5.0, n_trades)  # Small average profit
    
    df = pd.DataFrame({
        'pair': pair_names,
        'open_date': open_dates,
        'close_date': close_dates,
        'profit_abs': np.random.uniform(10, 1000, n_trades),
        'profit_ratio': profit_pct / 100,
        'trade_duration': durations_hours
    })
    
    return df


def save_test_dataset(data: Dict[str, pd.DataFrame], 
                     filepath: str, 
                     metadata: Dict = None):
    """Save test dataset with metadata"""
    dataset = {
        'metadata': metadata or {},
        'data': {}
    }
    
    for pair, df in data.items():
        # Convert DataFrame to dict with proper date serialization
        df_dict = df.to_dict(orient='records')
        # Convert datetime objects to strings
        for record in df_dict:
            if 'date' in record and isinstance(record['date'], pd.Timestamp):
                record['date'] = record['date'].isoformat()
        dataset['data'][pair] = df_dict
    
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)


def load_test_dataset(filepath: str) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Load test dataset and metadata"""
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    
    data = {}
    for pair, records in dataset['data'].items():
        df = pd.DataFrame(records)
        # Convert date strings back to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        data[pair] = df
    
    return data, dataset.get('metadata', {})


def create_performance_test_datasets():
    """Create the standard test datasets for performance benchmarking"""
    datasets_dir = Path("tests/performance/data")
    datasets_dir.mkdir(exist_ok=True)
    
    # Small dataset (1 year, 10 pairs)
    small_data = generate_trading_data(years=1, pairs=10, frequency='1h', seed=42)
    small_metadata = {
        'name': 'small_dataset',
        'description': '1 year of data, 10 trading pairs, 1h frequency',
        'years': 1,
        'pairs': 10,
        'frequency': '1h',
        'expected_performance_improvement': '15-20%'
    }
    save_test_dataset(small_data, datasets_dir / "small_dataset.json", small_metadata)
    
    # Medium dataset (3 years, 25 pairs) 
    medium_data = generate_trading_data(years=3, pairs=25, frequency='1h', seed=123)
    medium_metadata = {
        'name': 'medium_dataset',
        'description': '3 years of data, 25 trading pairs, 1h frequency', 
        'years': 3,
        'pairs': 25,
        'frequency': '1h',
        'expected_performance_improvement': '20-25%'
    }
    save_test_dataset(medium_data, datasets_dir / "medium_dataset.json", medium_metadata)
    
    # Large dataset (5 years, 50 pairs)
    large_data = generate_trading_data(years=5, pairs=50, frequency='1h', seed=456)
    large_metadata = {
        'name': 'large_dataset', 
        'description': '5 years of data, 50 trading pairs, 1h frequency',
        'years': 5,
        'pairs': 50,
        'frequency': '1h',
        'expected_performance_improvement': '25-30%'
    }
    save_test_dataset(large_data, datasets_dir / "large_dataset.json", large_metadata)
    
    # Trade parallelism test data
    trades_small = generate_trades_data(n_trades=500, seed=42)
    trades_medium = generate_trades_data(n_trades=2000, seed=123) 
    trades_large = generate_trades_data(n_trades=5000, seed=456)
    
    trades_small.to_pickle(datasets_dir / "trades_small.pkl")
    trades_medium.to_pickle(datasets_dir / "trades_medium.pkl")
    trades_large.to_pickle(datasets_dir / "trades_large.pkl")
    
    print(f"✅ Created performance test datasets in {datasets_dir}")
    return datasets_dir