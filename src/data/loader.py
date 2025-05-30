"""
Data loader module for financial news sentiment analysis.
This module provides functions for loading and basic preprocessing of financial news data.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any


def load_news_data(filepath: str) -> pd.DataFrame:
    """
    Load financial news data from a CSV file.
    
    Args:
        filepath: Path to the CSV file containing financial news data
        
    Returns:
        DataFrame containing the loaded data
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def preprocess_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on financial news data.
    
    Args:
        df: DataFrame containing financial news data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert date column to datetime if it exists
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Extract date components
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['weekday'] = df_processed['date'].dt.day_name()
        df_processed['hour'] = df_processed['date'].dt.hour
    
    # Add headline length if headline column exists
    if 'headline' in df_processed.columns:
        df_processed['headline_length'] = df_processed['headline'].apply(len)
        df_processed['word_count'] = df_processed['headline'].apply(lambda x: len(str(x).split()))
    
    return df_processed


def load_stock_data(directory: str, stock_symbols: Optional[list] = None) -> Dict[str, pd.DataFrame]:
    """
    Load historical stock price data for specified symbols.
    
    Args:
        directory: Directory containing stock data files
        stock_symbols: List of stock symbols to load, if None, load all available
        
    Returns:
        Dictionary mapping stock symbols to their respective DataFrames
    """
    stock_data = {}
    
    # If no specific symbols provided, try to load all CSV files in the directory
    if stock_symbols is None:
        files = [f for f in os.listdir(directory) if f.endswith('_historical_data.csv')]
        stock_symbols = [f.split('_')[0] for f in files]
    
    # Load each stock's data
    for symbol in stock_symbols:
        filepath = os.path.join(directory, f"{symbol}_historical_data.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Convert date column to datetime if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            stock_data[symbol] = df
            print(f"Loaded {symbol} data with shape: {df.shape}")
        else:
            print(f"Warning: No data file found for {symbol}")
    
    return stock_data


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where the CSV file will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
