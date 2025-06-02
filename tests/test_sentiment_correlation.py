import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from src.features.sentiment_correlation import (
    extract_sentiment,
    calculate_daily_returns,
    aggregate_daily_sentiment,
    calculate_correlation,
    calculate_lagged_correlation
)

class TestSentimentCorrelation(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.news_data = pd.DataFrame({
            'headline': [
                'Company X reports strong earnings',
                'Markets decline on economic fears',
                'Investors optimistic about tech sector',
                'Uncertainty looms over financial markets'
            ],
            'date': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3)
            ],
            'stock': ['AAPL', 'AAPL', 'AAPL', 'AAPL']
        })
        
        self.stock_data = pd.DataFrame({
            'Date': [
                datetime(2022, 12, 31),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4)
            ],
            'Close': [150.0, 152.0, 148.0, 151.0, 153.0]
        })
    
    def test_extract_sentiment(self):
        """Test sentiment extraction from headlines."""
        result = extract_sentiment(self.news_data)
        
        self.assertIn('sentiment', result.columns)
        self.assertIn('subjectivity', result.columns)
        
        self.assertTrue(all(result['sentiment'].between(-1, 1)))
        self.assertTrue(all(result['subjectivity'].between(0, 1)))
        
        self.assertGreater(result.loc[0, 'sentiment'], 0)  # Positive headline
        self.assertLess(result.loc[1, 'sentiment'], 0)  # Negative headline
    
    def test_calculate_daily_returns(self):
        """Test calculation of daily stock returns."""
        result = calculate_daily_returns(self.stock_data)
        
        self.assertIn('daily_return', result.columns)
        
        self.assertTrue(np.isnan(result.loc[0, 'daily_return']))
        self.assertAlmostEqual(result.loc[1, 'daily_return'], 0.01333, places=4)  # (152-150)/150
        self.assertAlmostEqual(result.loc[2, 'daily_return'], -0.02631, places=4)  # (148-152)/152
    
    def test_aggregate_daily_sentiment(self):
        """Test aggregation of daily sentiment."""
        news_with_sentiment = extract_sentiment(self.news_data)
        result = aggregate_daily_sentiment(news_with_sentiment)
        
        self.assertEqual(len(result), 3)  # 3 unique dates
        self.assertIn('date', result.columns)
        self.assertIn('sentiment', result.columns)
        self.assertIn('article_count', result.columns)
        
        jan2_row = result[result['date'] == datetime(2023, 1, 2)]
        self.assertEqual(jan2_row['article_count'].values[0], 2)
    
    def test_calculate_correlation(self):
        """Test correlation calculation between sentiment and returns."""
        news_with_sentiment = extract_sentiment(self.news_data)
        daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
        stock_with_returns = calculate_daily_returns(self.stock_data)
        
        corr, p_value = calculate_correlation(daily_sentiment, stock_with_returns)
        
        self.assertIsInstance(corr, float)
        self.assertIsInstance(p_value, float)
        self.assertTrue(-1 <= corr <= 1)
        self.assertTrue(0 <= p_value <= 1)
    
    def test_calculate_lagged_correlation(self):
        """Test lagged correlation calculations."""
        news_with_sentiment = extract_sentiment(self.news_data)
        daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
        stock_with_returns = calculate_daily_returns(self.stock_data)
        
        lagged_corrs = calculate_lagged_correlation(daily_sentiment, stock_with_returns, max_lag=2)
        
        self.assertIsInstance(lagged_corrs, dict)
        self.assertIn(0, lagged_corrs)
        self.assertIn(1, lagged_corrs)
        self.assertIn(2, lagged_corrs)
        
        for lag in range(3):
            self.assertEqual(len(lagged_corrs[lag]), 2)
            corr, p_value = lagged_corrs[lag]
            self.assertTrue(-1 <= corr <= 1)
            self.assertTrue(0 <= p_value <= 1)

if __name__ == '__main__':
    unittest.main()
