"""
NumPy-optimized versions of freqtrade functions
Pure NumPy vectorization approach for performance improvements without Numba dependency
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_market_change_optimized(
    data: Dict[str, pd.DataFrame], 
    column: str = "close", 
    min_date: Optional[datetime] = None
) -> float:
    """
    Optimized version of calculate_market_change using pure NumPy operations
    
    Performance improvements:
    - Vectorized data access
    - Direct array indexing instead of pandas .iloc
    - Reduced temporary object creation
    - Batch processing of all pairs
    
    Args:
        data: Dict of DataFrames, dict key should be pair
        column: Column in the original dataframes to use  
        min_date: Minimum date to consider for calculations
        
    Returns:
        float: Mean market change across all pairs
    """
    if not data:
        return 0.0
    
    pair_changes = []
    
    for pair, df in data.items():
        # Convert to numpy arrays for faster access
        values = df[column].values
        
        # Apply date filter if specified using numpy operations
        if min_date is not None:
            date_values = df["date"].values
            mask = date_values >= min_date
            values = values[mask]
            
        if len(values) == 0:
            logger.warning(f"Pair {pair} has no data after {min_date}.")
            continue
            
        # Use numpy operations for NaN handling and indexing
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            continue
            
        valid_values = values[valid_mask]
        if len(valid_values) < 2:
            continue
            
        # Direct array access - much faster than pandas .iloc
        start_price = valid_values[0]
        end_price = valid_values[-1]
        change = (end_price - start_price) / start_price
        pair_changes.append(change)
    
    if not pair_changes:
        return 0.0
        
    # Already using np.mean in original, keeping it
    return float(np.mean(pair_changes))


def combine_dataframes_by_column_optimized(
    data: Dict[str, pd.DataFrame], 
    column: str = "close"
) -> pd.DataFrame:
    """
    Optimized version of combine_dataframes_by_column using improved concat strategy
    
    Performance improvements:
    - Pre-filter columns to reduce memory usage
    - Batch set_index operations
    - Efficient column renaming
    - Reduced intermediate object creation
    
    Args:
        data: Dict of DataFrames, dict key should be pair
        column: Column in the original dataframes to use
        
    Returns:
        DataFrame with combined data
    """
    if not data:
        raise ValueError("No data provided.")
    
    # Optimized approach: prepare all series at once then concat
    series_list = []
    for pair, df in data.items():
        # Extract only the needed columns to reduce memory
        series = df.set_index("date")[column].rename(pair)
        series_list.append(series)
    
    # Single concat operation - more efficient than nested list comprehension
    df_comb = pd.concat(series_list, axis=1)
    
    return df_comb


def analyze_trade_parallelism_optimized(
    trades: pd.DataFrame, 
    timeframe: str
) -> pd.DataFrame:
    """
    Optimized version of analyze_trade_parallelism using custom binning algorithm
    
    Performance improvements:
    - Custom binning algorithm avoiding pandas resample overhead
    - Direct NumPy datetime operations for time bucketing
    - Vectorized trade period calculation
    - Efficient counting using NumPy operations
    
    Args:
        trades: Trades DataFrame
        timeframe: Timeframe used for backtest
        
    Returns:
        DataFrame with open-counts per time-period
    """
    from freqtrade.exchange import timeframe_to_resample_freq
    
    if trades.empty:
        return pd.DataFrame(columns=["open_trades"])
    
    timeframe_freq = timeframe_to_resample_freq(timeframe)
    
    # Convert to numpy datetime64 for faster operations
    open_dates = trades["open_date"].values.astype('datetime64[ns]')
    close_dates = trades["close_date"].values.astype('datetime64[ns]')
    
    # Calculate timeframe duration in nanoseconds
    freq_timedelta = pd.Timedelta(timeframe_freq)
    freq_ns = freq_timedelta.value
    
    # Find the overall time range and create bins
    min_time = open_dates.min()
    max_time = close_dates.max()
    
    # Align min_time to timeframe boundary (e.g., round to nearest hour for 1h)
    min_time_aligned = _align_to_timeframe_boundary(min_time, freq_ns)
    
    # Calculate number of bins needed
    total_duration_ns = int((max_time - min_time_aligned).astype('int64'))
    n_bins = int(np.ceil(total_duration_ns / freq_ns)) + 1
    
    if n_bins <= 0:
        return pd.DataFrame(columns=["open_trades"])
    
    # Create bin edges
    bin_edges = min_time_aligned + np.arange(n_bins + 1, dtype='int64') * freq_ns
    bin_centers = bin_edges[:-1]
    
    # Initialize count array
    trade_counts = np.zeros(n_bins, dtype=int)
    
    # For each trade, find which bins it spans and increment counts
    for open_date, close_date in zip(open_dates, close_dates):
        # Find first and last bin indices for this trade
        start_bin = max(0, int((open_date.astype('int64') - min_time_aligned.astype('int64')) // freq_ns))
        end_bin = min(n_bins - 1, int((close_date.astype('int64') - min_time_aligned.astype('int64')) // freq_ns))
        
        # Increment all bins this trade spans
        if start_bin < n_bins and end_bin >= 0:
            # Handle edge case where trade spans beyond our bin range
            start_bin = max(0, start_bin)
            end_bin = min(n_bins - 1, end_bin)
            trade_counts[start_bin:end_bin + 1] += 1
    
    # Remove empty bins at the end
    last_nonzero = np.where(trade_counts > 0)[0]
    if len(last_nonzero) == 0:
        return pd.DataFrame(columns=["open_trades"])
    
    last_idx = last_nonzero[-1] + 1
    trade_counts = trade_counts[:last_idx]
    bin_centers = bin_centers[:last_idx]
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'open_trades': trade_counts
    }, index=pd.Index(pd.to_datetime(bin_centers), name='date'))
    
    return result_df


def _align_to_timeframe_boundary(timestamp: np.datetime64, freq_ns: int) -> np.datetime64:
    """
    Align timestamp to timeframe boundary (e.g., round down to nearest hour for 1h timeframe)
    """
    ts_ns = int(timestamp.astype('int64'))
    aligned_ns = (ts_ns // freq_ns) * freq_ns
    return np.datetime64(aligned_ns, 'ns')


def calculate_max_drawdown_optimized(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_abs",
    starting_balance: float = 0,
    relative: bool = False,
):
    """
    Optimized version of calculate_max_drawdown using pure NumPy operations
    
    Performance improvements:
    - Vectorized cumulative operations
    - Direct NumPy array operations instead of pandas
    - Single-pass calculation for all metrics
    - Optimized max/min finding
    
    Args:
        trades: DataFrame containing trades
        date_col: Column to use for dates
        value_col: Column to use for values  
        starting_balance: Portfolio starting balance
        relative: Use relative drawdown for max calculation
        
    Returns:
        DrawDownResult object with drawdown metrics
    """
    from freqtrade.data.metrics import DrawDownResult
    
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")

    # Sort by date and extract values as numpy arrays
    sorted_trades = trades.sort_values(date_col).reset_index(drop=True)
    values = sorted_trades[value_col].values.astype(np.float64)
    dates = sorted_trades[date_col].values
    
    # Vectorized cumulative sum
    cumulative = np.cumsum(values)
    
    # Calculate running maximum using numpy operations
    high_values = np.maximum(0, np.maximum.accumulate(cumulative))
    
    # Calculate drawdowns
    drawdowns = cumulative - high_values
    
    # Calculate relative drawdowns
    if starting_balance:
        cumulative_balance = starting_balance + cumulative
        max_balance = starting_balance + high_values
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown_relative = np.where(
                max_balance != 0,
                (max_balance - cumulative_balance) / max_balance,
                0
            )
    else:
        # Fallback calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown_relative = np.where(
                high_values != 0,
                (high_values - cumulative) / high_values,
                0
            )
    
    # Find maximum drawdown index using numpy
    if relative:
        idxmin = np.argmax(drawdown_relative)
    else:
        idxmin = np.argmin(drawdowns)
    
    # Find high index efficiently
    high_idx = np.argmax(high_values[:idxmin + 1])
    
    # Extract results
    high_date = dates[high_idx]
    low_date = dates[idxmin]
    high_val = cumulative[high_idx]
    low_val = cumulative[idxmin]
    max_drawdown_rel = drawdown_relative[idxmin]
    
    # Calculate current drawdown efficiently
    if len(high_values) > 1:
        current_high_idx = np.argmax(high_values[:-1])
    else:
        current_high_idx = 0
        
    current_high_date = dates[current_high_idx]
    current_high_value = high_values[-1]
    current_cumulative = cumulative[-1]
    current_drawdown_abs = current_high_value - current_cumulative
    current_drawdown_relative = drawdown_relative[-1]
    
    return DrawDownResult(
        # Max drawdown
        drawdown_abs=abs(drawdowns[idxmin]),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel,
        # Current drawdown
        current_high_date=current_high_date,
        current_high_value=current_high_value,
        current_drawdown_abs=current_drawdown_abs,
        current_relative_account_drawdown=current_drawdown_relative,
    )


def calculate_sharpe_optimized(trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float) -> float:
    """
    Optimized Sharpe ratio calculation using NumPy operations
    
    Performance improvements:
    - Direct numpy statistical operations
    - Vectorized calculations
    - Single pass through data
    
    Args:
        trades: DataFrame with profit data
        min_date: Minimum date (used for filtering)
        max_date: Maximum date (used for filtering) 
        starting_balance: Starting balance (unused in calculation)
        
    Returns:
        Sharpe ratio
    """
    # Quick exit conditions
    if len(trades) == 0 or min_date is None or max_date is None or min_date == max_date:
        return 0
    
    # Use numpy vectorized operations for better performance
    profit_abs = trades["profit_abs"].values.astype(np.float64)
    total_profit = np.sum(profit_abs) / starting_balance
    days_period = max(1, (max_date - min_date).days)
    
    # Vectorized operations
    expected_returns_mean = total_profit / days_period
    up_stdev = np.std(profit_abs / starting_balance)
    
    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = -100
        
    return sharp_ratio


def calculate_sortino_optimized(trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float) -> float:
    """
    Optimized Sortino ratio calculation using NumPy operations
    
    Performance improvements:
    - Vectorized downside deviation calculation
    - Direct numpy operations
    - Single pass computation
    
    Args:
        trades: DataFrame with profit data
        min_date: Minimum date (used for filtering)
        max_date: Maximum date (used for filtering)
        starting_balance: Starting balance (unused in calculation)
        
    Returns:
        Sortino ratio
    """
    # Quick exit conditions
    if len(trades) == 0 or min_date is None or max_date is None or min_date == max_date:
        return 0
        
    # Use numpy vectorized operations for better performance 
    profit_abs = trades["profit_abs"].values.astype(np.float64)
    total_profit = np.sum(profit_abs) / starting_balance
    days_period = max(1, (max_date - min_date).days)
    
    expected_returns_mean = total_profit / days_period
    
    # Calculate downside standard deviation for negative profits only - vectorized
    negative_mask = profit_abs < 0
    if np.any(negative_mask):
        down_stdev = np.std(profit_abs[negative_mask] / starting_balance)
    else:
        down_stdev = 0
    
    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # Define high (negative) sortino ratio to be clear that this is NOT optimal.
        sortino_ratio = -100
        
    return sortino_ratio


def calculate_calmar_optimized(trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float) -> float:
    """
    Optimized Calmar ratio calculation
    
    Performance improvements:
    - Reuse optimized max drawdown calculation
    - Direct computation without intermediate dataframes
    
    Args:
        trades: DataFrame with profit data
        min_date: Minimum date (used for filtering)
        max_date: Maximum date (used for filtering)  
        starting_balance: Starting balance for calculations
        
    Returns:
        Calmar ratio
    """
    # Quick exit conditions
    if len(trades) == 0 or min_date is None or max_date is None or min_date == max_date:
        return 0
        
    # Use optimized max drawdown calculation with numpy operations
    try:
        drawdown_result = calculate_max_drawdown_optimized(
            trades, starting_balance=starting_balance, relative=True
        )
        max_drawdown = drawdown_result.relative_account_drawdown
    except (ValueError, ZeroDivisionError):
        return 0
        
    # Use numpy operations for speed
    profit_abs = trades["profit_abs"].values.astype(np.float64)
    total_profit = np.sum(profit_abs) / starting_balance
    days_period = max(1, (max_date - min_date).days)
    
    expected_returns_mean = total_profit / days_period * 100
    
    if max_drawdown != 0:
        import math
        calmar_ratio = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        # Define high (negative) calmar ratio to be clear that this is NOT optimal.
        calmar_ratio = -100
        
    return calmar_ratio


# Fallback mechanism - automatically use optimized versions when available
def get_optimized_function(function_name: str):
    """
    Get the optimized version of a function if available, otherwise return None
    This allows for gradual rollout and easy fallback
    """
    optimized_functions = {
        'calculate_market_change': calculate_market_change_optimized,
        'combine_dataframes_by_column': combine_dataframes_by_column_optimized, 
        'analyze_trade_parallelism': analyze_trade_parallelism_optimized,
        'calculate_max_drawdown': calculate_max_drawdown_optimized,
        'calculate_sharpe': calculate_sharpe_optimized,
        'calculate_sortino': calculate_sortino_optimized,
        'calculate_calmar': calculate_calmar_optimized,
        'get_ohlcv_as_lists': get_ohlcv_as_lists_optimized,
        'get_ohlcv_as_lists_vectorized': get_ohlcv_as_lists_vectorized_alternative
    }
    return optimized_functions.get(function_name)


def get_ohlcv_as_lists_optimized(
    processed: dict[str, pd.DataFrame], 
    strategy,
    dataprovider, 
    timerange,
    required_startup: int,
    progress_callback=None
) -> dict[str, list]:
    """
    Optimized version of Backtesting._get_ohlcv_as_lists using vectorized operations
    
    Performance improvements:
    - Vectorized signal shifting instead of column-by-column operations
    - Direct NumPy array processing to minimize DataFrame operations  
    - In-place operations where safe to reduce memory allocations
    - Batch processing of similar column types
    - Eliminated unnecessary DataFrame copying
    
    Args:
        processed: Dictionary of {pair: DataFrame} with OHLCV data
        strategy: Trading strategy instance for signal generation
        dataprovider: DataProvider instance for caching
        timerange: TimeRange for trimming data
        required_startup: Number of startup candles to skip
        progress_callback: Optional progress update callback
        
    Returns:
        Dictionary of {pair: list} where each list contains OHLCV + signal data
    """
    # Import here to avoid circular dependencies
    from freqtrade.data.converter.converter import trim_dataframe
    
    # Headers definition (matching backtesting.py)
    HEADERS = [
        "date", "open", "high", "low", "close",
        "enter_long", "exit_long", "enter_short", "exit_short", 
        "enter_tag", "exit_tag"
    ]
    
    data = {}
    total_pairs = len(processed)
    
    if progress_callback:
        progress_callback("Converting OHLCV data", 0, total_pairs)
    
    for i, pair in enumerate(processed.keys()):
        pair_data = processed[pair]
        
        if not pair_data.empty:
            # Step 1: Clean up columns in-place for memory efficiency
            columns_to_drop = HEADERS[5:] + ["buy", "sell"] 
            existing_columns = [col for col in columns_to_drop if col in pair_data.columns]
            if existing_columns:
                pair_data.drop(existing_columns, axis=1, errors="ignore", inplace=True)
        
        # Step 2: Apply strategy signals (unchanged - this is core business logic)
        df_analyzed = strategy.ft_advise_signals(pair_data, {"pair": pair})
        
        # Step 3: Cache dataframe (unchanged - needed for strategy callbacks)
        dataprovider._set_cached_df(
            pair, strategy.timeframe, df_analyzed, {"candle_type_def": "spot"}
        )
        
        # Step 4: Trim startup period (unchanged - needed for correctness)
        df_analyzed = trim_dataframe(
            df_analyzed, timerange, startup_candles=required_startup
        )
        
        # Update the processed dict reference for memory cleanup
        processed[pair] = df_analyzed
        
        if df_analyzed.empty:
            data[pair] = []
            continue
            
        # Step 5: OPTIMIZED - Vectorized signal processing
        # Instead of looping through columns individually, batch process by type
        
        # Ensure all required signal columns exist
        signal_cols = ["enter_long", "exit_long", "enter_short", "exit_short"]
        tag_cols = ["enter_tag", "exit_tag"] 
        
        for col in signal_cols:
            if col not in df_analyzed.columns:
                df_analyzed[col] = 0
                
        for col in tag_cols:
            if col not in df_analyzed.columns:
                df_analyzed[col] = None
        
        # OPTIMIZATION: Direct numpy array processing for maximum speed
        # Convert to numpy array once, process in-place, convert back
        array_data = df_analyzed[HEADERS].values
        n_rows, n_cols = array_data.shape
        
        if n_rows > 0:
            # Vectorized shifting: Move all signal/tag data down by 1 row
            # This is much faster than individual column operations
            
            # Shift signals (columns 5-8: enter_long, exit_long, enter_short, exit_short)
            array_data[1:, 5:9] = array_data[:-1, 5:9]
            array_data[0, 5:9] = 0  # Fill first row with zeros
            
            # Shift tags (columns 9-10: enter_tag, exit_tag)  
            array_data[1:, 9:11] = array_data[:-1, 9:11]
            array_data[0, 9:11] = None  # Fill first row with None
            
            # Drop first row (equivalent to previous shift + drop operation)
            array_data = array_data[1:]  # Direct array slicing - very fast
            
            # Convert to list format (required for downstream backtesting performance)
            data[pair] = array_data.tolist()
        else:
            data[pair] = []
        
        # Update progress
        if progress_callback:
            progress_callback("Converting OHLCV data", i + 1, total_pairs)
    
    return data


def get_ohlcv_as_lists_vectorized_alternative(
    processed: dict[str, pd.DataFrame], 
    strategy,
    dataprovider, 
    timerange,
    required_startup: int,
    progress_callback=None
) -> dict[str, list]:
    """
    Alternative vectorized approach using pandas operations optimized for speed
    This version uses pandas vectorized operations instead of raw NumPy for better compatibility
    """
    from freqtrade.data.converter.converter import trim_dataframe
    
    HEADERS = [
        "date", "open", "high", "low", "close",
        "enter_long", "exit_long", "enter_short", "exit_short", 
        "enter_tag", "exit_tag"
    ]
    
    data = {}
    total_pairs = len(processed)
    
    for i, pair in enumerate(processed.keys()):
        pair_data = processed[pair]
        
        if not pair_data.empty:
            # Efficient column cleanup
            columns_to_drop = [col for col in (HEADERS[5:] + ["buy", "sell"]) if col in pair_data.columns]
            if columns_to_drop:
                pair_data = pair_data.drop(columns_to_drop, axis=1)
        
        # Strategy signals (unchanged)
        df_analyzed = strategy.ft_advise_signals(pair_data, {"pair": pair})
        
        # Cache dataframe (unchanged)
        dataprovider._set_cached_df(
            pair, strategy.timeframe, df_analyzed, {"candle_type_def": "spot"}
        )
        
        # Trim startup period (unchanged)
        df_analyzed = trim_dataframe(
            df_analyzed, timerange, startup_candles=required_startup
        )
        
        if df_analyzed.empty:
            data[pair] = []
            continue
        
        # Make a proper copy to avoid SettingWithCopyWarning
        df_analyzed = df_analyzed.copy()
        
        # Ensure all signal columns exist with correct defaults
        for col in ["enter_long", "exit_long", "enter_short", "exit_short"]:
            if col not in df_analyzed.columns:
                df_analyzed[col] = 0
                
        for col in ["enter_tag", "exit_tag"]:
            if col not in df_analyzed.columns:
                df_analyzed[col] = None
        
        # OPTIMIZED: Vectorized pandas operations
        # Group signals and tags for batch processing
        signal_columns = ["enter_long", "exit_long", "enter_short", "exit_short"]
        tag_columns = ["enter_tag", "exit_tag"]
        
        # Vectorized shift for signals - single operation instead of 4 separate ones
        df_analyzed.loc[:, signal_columns] = (
            df_analyzed[signal_columns]
            .shift(1)
            .fillna(0)
        )
        
        # Vectorized shift for tags - single operation instead of 2 separate ones  
        df_analyzed.loc[:, tag_columns] = (
            df_analyzed[tag_columns]
            .shift(1)
        )
        
        # Fill NaN values with None for tag columns
        for col in tag_columns:
            df_analyzed.loc[:, col] = df_analyzed[col].where(df_analyzed[col].notna(), None)
        
        # Drop first row after shifting (more efficient than head/drop combination)
        df_analyzed = df_analyzed.iloc[1:].copy() if len(df_analyzed) > 1 else df_analyzed.iloc[0:0]
        
        # Convert to list (unchanged - required for downstream performance)
        data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        
        if progress_callback:
            progress_callback("Converting OHLCV data", i + 1, total_pairs)
    
    return data


# Performance monitoring decorator
def performance_monitor(original_func):
    """
    Decorator to monitor performance improvements
    Can be used during development and testing
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.perf_counter()
        result = original_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        function_name = original_func.__name__
        execution_time = (end_time - start_time) * 1000  # ms
        
        # Log performance metrics (can be disabled in production)
        logger.debug(f"{function_name} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper