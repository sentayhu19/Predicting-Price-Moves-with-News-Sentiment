import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union


def add_moving_averages(df: pd.DataFrame, 
                       periods: List[int] = [5, 10, 20, 50, 200],
                       price_col: str = 'Close') -> pd.DataFrame:
    """
    Add simple moving averages to the DataFrame.
    
    Args:
        df: DataFrame with price data
        periods: List of periods for calculating MAs
        price_col: Column to use for price data
        
    Returns:
        DataFrame with added MA columns
    """
    result = df.copy()
    
    for period in periods:
        result[f'sma_{period}'] = result[price_col].rolling(window=period).mean()
        # Exponential moving average
        result[f'ema_{period}'] = result[price_col].ewm(span=period, adjust=False).mean()
    
    return result


def add_rsi(df: pd.DataFrame, 
          period: int = 14,
          price_col: str = 'Close') -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) to the DataFrame.
    
    Args:
        df: DataFrame with price data
        period: Period for calculating RSI
        price_col: Column to use for price data
        
    Returns:
        DataFrame with added RSI column
    """
    result = df.copy()
    
    # Calculate price changes
    delta = result[price_col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    return result


def add_bollinger_bands(df: pd.DataFrame,
                       period: int = 20,
                       std_dev: float = 2.0,
                       price_col: str = 'Close') -> pd.DataFrame:
    """
    Add Bollinger Bands to the DataFrame.
    
    Args:
        df: DataFrame with price data
        period: Period for calculating the moving average
        std_dev: Number of standard deviations for the bands
        price_col: Column to use for price data
        
    Returns:
        DataFrame with added Bollinger Bands columns
    """
    result = df.copy()
    
    # Calculate the middle band (SMA)
    result['bb_middle'] = result[price_col].rolling(window=period).mean()
    
    # Calculate the standard deviation
    rolling_std = result[price_col].rolling(window=period).std()
    
    # Calculate upper and lower bands
    result['bb_upper'] = result['bb_middle'] + (rolling_std * std_dev)
    result['bb_lower'] = result['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate Bollinger Band width
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    
    return result


def add_macd(df: pd.DataFrame,
           fast_period: int = 12,
           slow_period: int = 26,
           signal_period: int = 9,
           price_col: str = 'Close') -> pd.DataFrame:
    """
    Add Moving Average Convergence Divergence (MACD) to the DataFrame.
    
    Args:
        df: DataFrame with price data
        fast_period: Period for the fast EMA
        slow_period: Period for the slow EMA
        signal_period: Period for the signal line
        price_col: Column to use for price data
        
    Returns:
        DataFrame with added MACD columns
    """
    result = df.copy()
    
    # Calculate the fast and slow EMAs
    ema_fast = result[price_col].ewm(span=fast_period, adjust=False).mean()
    ema_slow = result[price_col].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    result['macd_line'] = ema_fast - ema_slow
    
    # Calculate the signal line
    result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate the histogram
    result['macd_histogram'] = result['macd_line'] - result['macd_signal']
    
    return result


def add_stochastic(df: pd.DataFrame,
                 k_period: int = 14,
                 d_period: int = 3) -> pd.DataFrame:
    """
    Add Stochastic Oscillator to the DataFrame.
    
    Args:
        df: DataFrame with price data
        k_period: Period for %K line
        d_period: Period for %D line
        
    Returns:
        DataFrame with added Stochastic Oscillator columns
    """
    result = df.copy()
    
    # Calculate %K
    # Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    low_min = result['Low'].rolling(window=k_period).min()
    high_max = result['High'].rolling(window=k_period).max()
    
    result['stoch_k'] = 100 * ((result['Close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (3-day SMA of %K)
    result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
    
    return result


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to the DataFrame.
    
    Args:
        df: DataFrame with price data
        period: Period for calculating ATR
        
    Returns:
        DataFrame with added ATR column
    """
    result = df.copy()
    
    # Calculate True Range
    result['tr1'] = abs(result['High'] - result['Low'])
    result['tr2'] = abs(result['High'] - result['Close'].shift())
    result['tr3'] = abs(result['Low'] - result['Close'].shift())
    
    result['true_range'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR
    result[f'atr_{period}'] = result['true_range'].rolling(window=period).mean()
    
    # Drop temporary columns
    result = result.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
    
    return result


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based indicators to the DataFrame.
    
    Args:
        df: DataFrame with price and volume data
        
    Returns:
        DataFrame with added volume indicator columns
    """
    result = df.copy()
    
    # Check if Volume column exists
    if 'Volume' not in result.columns:
        print("Warning: Volume column not found. Volume indicators will not be calculated.")
        return result
    
    # On-Balance Volume (OBV)
    result['price_change'] = result['Close'].diff()
    result['obv'] = 0
    
    # Calculate OBV
    for i in range(1, len(result)):
        if result['price_change'].iloc[i] > 0:
            result['obv'].iloc[i] = result['obv'].iloc[i-1] + result['Volume'].iloc[i]
        elif result['price_change'].iloc[i] < 0:
            result['obv'].iloc[i] = result['obv'].iloc[i-1] - result['Volume'].iloc[i]
        else:
            result['obv'].iloc[i] = result['obv'].iloc[i-1]
    
    # Volume-Weighted Average Price (VWAP)
    result['vwap'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
    
    # Price-Volume Trend (PVT)
    result['pvt'] = ((result['Close'].diff() / result['Close'].shift()) * result['Volume']).cumsum()
    
    # Drop temporary columns
    result = result.drop(['price_change'], axis=1)
    
    return result


def add_all_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all simple technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical indicators
    """
    result = df.copy()
    
    try:
        print("Adding simple technical indicators...")
        
        # Add moving averages
        print("  Adding moving averages...")
        result = add_moving_averages(result)
        
        # Add RSI
        print("  Adding RSI...")
        result = add_rsi(result)
        
        # Add Bollinger Bands
        print("  Adding Bollinger Bands...")
        result = add_bollinger_bands(result)
        
        # Add MACD
        print("  Adding MACD...")
        result = add_macd(result)
        
        # Add Stochastic Oscillator
        print("  Adding Stochastic Oscillator...")
        result = add_stochastic(result)
        
        # Add ATR
        print("  Adding ATR...")
        result = add_atr(result)
        
        # Add volume indicators
        print("  Adding volume indicators...")
        result = add_volume_indicators(result)
        
        print("All simple indicators added successfully!")
        return result
    except Exception as e:
        print(f"Error adding simple indicators: {e}")
        # If there's an error, return the original DataFrame
        return df


def generate_simple_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on simple indicators.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with trading signals
    """
    result = df.copy()
    
    try:
        # Initialize signals
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        
        # RSI signals
        if 'rsi_14' in result.columns:
            # Buy when RSI crosses above 30 (oversold)
            result.loc[(result['rsi_14'] > 30) & (result['rsi_14'].shift(1) <= 30), 'buy_signal'] = 1
            # Sell when RSI crosses below 70 (overbought)
            result.loc[(result['rsi_14'] < 70) & (result['rsi_14'].shift(1) >= 70), 'sell_signal'] = 1
        
        # MACD signals
        if 'macd_line' in result.columns and 'macd_signal' in result.columns:
            # Buy when MACD line crosses above signal line
            result.loc[(result['macd_line'] > result['macd_signal']) & 
                      (result['macd_line'].shift(1) <= result['macd_signal'].shift(1)), 'buy_signal'] = 1
            # Sell when MACD line crosses below signal line
            result.loc[(result['macd_line'] < result['macd_signal']) & 
                      (result['macd_line'].shift(1) >= result['macd_signal'].shift(1)), 'sell_signal'] = 1
        
        # Bollinger Bands signals
        if 'bb_lower' in result.columns and 'bb_upper' in result.columns:
            # Buy when price crosses above lower band
            result.loc[(result['Close'] > result['bb_lower']) & 
                      (result['Close'].shift(1) <= result['bb_lower'].shift(1)), 'buy_signal'] = 1
            # Sell when price crosses below upper band
            result.loc[(result['Close'] < result['bb_upper']) & 
                      (result['Close'].shift(1) >= result['bb_upper'].shift(1)), 'sell_signal'] = 1
        
        # Moving Average crossovers
        if 'sma_20' in result.columns and 'sma_50' in result.columns:
            # Buy when 20-day SMA crosses above 50-day SMA (golden cross)
            result.loc[(result['sma_20'] > result['sma_50']) & 
                      (result['sma_20'].shift(1) <= result['sma_50'].shift(1)), 'buy_signal'] = 1
            # Sell when 20-day SMA crosses below 50-day SMA (death cross)
            result.loc[(result['sma_20'] < result['sma_50']) & 
                      (result['sma_20'].shift(1) >= result['sma_50'].shift(1)), 'sell_signal'] = 1
        
        # Combine signals
        result['signal_strength'] = result['buy_signal'] - result['sell_signal']
        
        print("Trading signals generated successfully")
        return result
    except Exception as e:
        print(f"Error generating signals: {e}")
        return result
