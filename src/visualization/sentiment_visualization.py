import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import os


def plot_sentiment_returns_scatter(merged_df: pd.DataFrame, 
                                  sentiment_col: str = 'avg_sentiment',
                                  returns_col: str = 'daily_return',
                                  date_col: str = 'date',
                                  title: str = 'Sentiment vs. Stock Returns',
                                  output_path: Optional[str] = None) -> None:
    """
    Create a scatter plot of sentiment scores vs. stock returns.
    
    Args:
        merged_df: DataFrame with sentiment and returns data
        sentiment_col: Column containing sentiment scores
        returns_col: Column containing stock returns
        date_col: Column containing dates
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(merged_df[sentiment_col], merged_df[returns_col], alpha=0.6)
    
    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged_df[sentiment_col], merged_df[returns_col]
    )
    x = merged_df[sentiment_col]
    plt.plot(x, intercept + slope * x, 'r', 
             label=f'Trend line (r={r_value:.3f}, p={p_value:.3f})')
    
    # Add labels and title
    plt.xlabel('Sentiment Score')
    plt.ylabel('Stock Return (%)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_sentiment_returns_time_series(merged_df: pd.DataFrame,
                                      sentiment_col: str = 'avg_sentiment',
                                      returns_col: str = 'daily_return',
                                      date_col: str = 'date',
                                      title: str = 'Sentiment and Returns Over Time',
                                      output_path: Optional[str] = None) -> None:
    """
    Create a time series plot of sentiment scores and stock returns.
    
    Args:
        merged_df: DataFrame with sentiment and returns data
        sentiment_col: Column containing sentiment scores
        returns_col: Column containing stock returns
        date_col: Column containing dates
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
    """
    plt.figure(figsize=(12, 8))
    
    # Ensure data is sorted by date
    df = merged_df.sort_values(by=date_col)
    
    # Create two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot sentiment on first axis
    ax1.plot(df[date_col], df[sentiment_col], 'b-', label='Sentiment')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot returns on second axis
    ax2.plot(df[date_col], df[returns_col], 'g-', label='Returns')
    ax2.set_ylabel('Stock Return (%)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add title and format
    plt.title(title)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_lagged_correlations(correlation_results: Dict,
                            max_lag: int = 5,
                            title: str = 'Lagged Correlations',
                            output_path: Optional[str] = None) -> None:
    """
    Create a bar chart of lagged correlations.
    
    Args:
        correlation_results: Dictionary containing correlation results with lagged_correlations
        max_lag: Maximum number of lags to plot
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
    """
    # Extract lagged correlations
    lagged_correlations = correlation_results['lagged_correlations']
    
    # Prepare data for plotting
    lags = []
    sentiment_leading_returns = []
    returns_leading_sentiment = []
    p_values_sent_lead = []
    p_values_ret_lead = []
    
    for lag_data in lagged_correlations[:max_lag]:
        lags.append(lag_data['lag'])
        sentiment_leading_returns.append(lag_data['sentiment_leading_returns_corr'])
        returns_leading_sentiment.append(lag_data['returns_leading_sentiment_corr'])
        p_values_sent_lead.append(lag_data['sentiment_leading_returns_p'])
        p_values_ret_lead.append(lag_data['returns_leading_sentiment_p'])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set positions of bars on X axis
    r1 = np.arange(len(lags))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    sent_bars = plt.bar(r1, sentiment_leading_returns, width=bar_width, 
                      edgecolor='grey', label='Sentiment→Returns')
    ret_bars = plt.bar(r2, returns_leading_sentiment, width=bar_width,
                     edgecolor='grey', label='Returns→Sentiment')
    
    # Highlight significant bars
    for i, (bar, p) in enumerate(zip(sent_bars, p_values_sent_lead)):
        if p < 0.05:
            bar.set_color('green')
        else:
            bar.set_color('lightblue')
            
    for i, (bar, p) in enumerate(zip(ret_bars, p_values_ret_lead)):
        if p < 0.05:
            bar.set_color('red')
        else:
            bar.set_color('salmon')
    
    # Add labels and title
    plt.xlabel('Lag (Days)')
    plt.ylabel('Correlation Coefficient')
    plt.title(title)
    plt.xticks([r + bar_width/2 for r in range(len(lags))], lags)
    plt.grid(axis='y', alpha=0.3)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Significant Sentiment→Returns (p<0.05)'),
        Patch(facecolor='lightblue', label='Non-significant Sentiment→Returns'),
        Patch(facecolor='red', label='Significant Returns→Sentiment (p<0.05)'),
        Patch(facecolor='salmon', label='Non-significant Returns→Sentiment')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_sentiment_distribution(news_df: pd.DataFrame,
                               sentiment_col: str = 'polarity',
                               title: str = 'Distribution of Headline Sentiment',
                               output_path: Optional[str] = None) -> None:
    """
    Create a histogram of sentiment scores.
    
    Args:
        news_df: DataFrame with sentiment data
        sentiment_col: Column containing sentiment scores
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(news_df[sentiment_col], kde=True)
    
    # Add labels and title
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    # Add mean and median lines
    mean = news_df[sentiment_col].mean()
    median = news_df[sentiment_col].median()
    plt.axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    plt.legend()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def create_sentiment_stock_dashboard(correlation_results: Dict,
                                    ticker: str,
                                    output_dir: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard of sentiment-stock correlation visualizations.
    
    Args:
        correlation_results: Dictionary containing correlation results
        ticker: Stock ticker symbol
        output_dir: Directory to save plots (if None, displays instead)
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    merged_df = correlation_results['data']
    
    # 1. Scatter plot of sentiment vs returns
    scatter_path = os.path.join(output_dir, f"{ticker}_sentiment_returns_scatter.png") if output_dir else None
    plot_sentiment_returns_scatter(
        merged_df, 
        title=f'Sentiment vs. Daily Returns for {ticker}',
        output_path=scatter_path
    )
    
    # 2. Time series plot
    time_series_path = os.path.join(output_dir, f"{ticker}_sentiment_returns_time_series.png") if output_dir else None
    plot_sentiment_returns_time_series(
        merged_df,
        title=f'Sentiment and Returns Over Time for {ticker}',
        output_path=time_series_path
    )
    
    # 3. Lagged correlations
    lagged_path = os.path.join(output_dir, f"{ticker}_lagged_correlations.png") if output_dir else None
    plot_lagged_correlations(
        correlation_results,
        title=f'Lagged Correlations for {ticker}',
        output_path=lagged_path
    )
    
    if output_dir:
        print(f"Dashboard visualizations saved to {output_dir}")


def compare_multiple_stocks(results_dict: Dict[str, Dict],
                          output_dir: Optional[str] = None) -> None:
    """
    Create comparison visualizations for multiple stocks.
    
    Args:
        results_dict: Dictionary mapping tickers to correlation results
        output_dir: Directory to save plots (if None, displays instead)
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract tickers and correlations
    tickers = list(results_dict.keys())
    correlations = [results_dict[ticker]['correlation'] for ticker in tickers]
    p_values = [results_dict[ticker]['p_value'] for ticker in tickers]
    
    # 1. Bar chart of correlations
    plt.figure(figsize=(12, 6))
    
    # Set colors based on significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    
    # Create bars
    bars = plt.bar(tickers, correlations, color=colors)
    
    # Add correlation values on top of bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + (0.01 if height >= 0 else -0.03),
                f'{corr:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top')
    
    # Add labels and formatting
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlation between News Sentiment and Stock Returns')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.grid(axis='y', alpha=0.3)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Statistically Significant (p<0.05)'),
        Patch(facecolor='gray', label='Not Significant')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Save or display
    if output_dir:
        plt.savefig(os.path.join(output_dir, "stock_correlation_comparison.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # 2. Heatmap of correlations at different lags
    # Prepare data for heatmap
    max_lag = 5
    heatmap_data = np.zeros((len(tickers), max_lag))
    
    for i, ticker in enumerate(tickers):
        lagged_correlations = results_dict[ticker]['lagged_correlations']
        for j in range(max_lag):
            if j < len(lagged_correlations):
                heatmap_data[i, j] = lagged_correlations[j]['sentiment_leading_returns_corr']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0,
               xticklabels=[f'Lag {i+1}' for i in range(max_lag)],
               yticklabels=tickers)
    
    plt.title('Sentiment Leading Returns Correlation by Lag')
    plt.xlabel('Lag (Days)')
    plt.ylabel('Stock Ticker')
    
    # Save or display
    if output_dir:
        plt.savefig(os.path.join(output_dir, "lagged_correlation_heatmap.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    if output_dir:
        print(f"Comparison visualizations saved to {output_dir}")
