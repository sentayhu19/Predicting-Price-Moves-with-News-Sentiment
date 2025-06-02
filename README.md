# Predicting Price Moves with News Sentiment

A comprehensive project for analyzing financial news sentiment and its correlation with stock price movements.

## Overview

This project focuses on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. The analysis includes sentiment analysis of news headlines and correlation analysis between sentiment and stock price fluctuations.

## Business Objective

Enhance predictive analytics capabilities to boost financial forecasting accuracy and operational efficiency through advanced data analysis. The project has two main focuses:

1. **Sentiment Analysis**: Quantifying the tone and sentiment in financial news headlines using NLP techniques.
2. **Correlation Analysis**: Establishing statistical correlations between news sentiment and corresponding stock price movements.

## Dataset

The Financial News and Stock Price Integration Dataset (FNSPID) includes:

- **headline**: Article release headline/title
- **url**: Direct link to the full news article
- **publisher**: Author/creator of article
- **date**: Publication date and time (UTC-4 timezone)
- **stock**: Stock ticker symbol (e.g., AAPL for Apple)

## Project Structure

```
├── .vscode/
│   └── settings.json        # VS Code settings
├── .github/
│   └── workflows
│       ├── unittests.yml    # CI/CD pipeline configuration
├── .gitignore               # Git ignore file
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   ├── models/              # Model training and evaluation
│   └── visualization/       # Data visualization
├── notebooks/               # Jupyter notebooks
│   ├── __init__.py
│   └── README.md            # Notebook documentation
├── tests/                   # Unit tests
│   ├── __init__.py
└── scripts/                 # Utility scripts
    ├── __init__.py
```

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/sentayhu19/Predicting-Price-Moves-with-News-Sentiment.git
cd Predicting-Price-Moves-with-News-Sentiment
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Features

### Task 3: News Sentiment and Stock Correlation Analysis

A complete implementation of the correlation analysis between news headline sentiment and stock price movements. This feature includes:

- **Date Alignment**: Normalizes dates between news and stock datasets for accurate correlation
- **Sentiment Analysis**: Uses TextBlob to analyze sentiment in financial headlines
- **Stock Returns Calculation**: Computes daily percentage changes in stock closing prices
- **Aggregation**: Combines multiple headlines per day into a single sentiment score
- **Correlation Analysis**: Calculates Pearson correlation coefficient between sentiment and stock returns
- **Lagged Analysis**: Examines time-delayed relationships between sentiment and price movements
- **Visualization**: Provides comprehensive visualizations for analyzing the sentiment-price relationship

### Key Modules

- `src/features/sentiment_correlation.py`: Core implementation for sentiment analysis and correlation
- `src/visualization/sentiment_visualization.py`: Visualization tools for sentiment-price relationships
- `src/examples/sentiment_correlation_example.py`: Example script demonstrating the full analysis pipeline

## Usage

### Running the Sentiment Correlation Analysis

```bash
# Navigate to the project directory
cd Predicting-Price-Moves-with-News-Sentiment

# Run the example script
python src/examples/sentiment_correlation_example.py
```

This will:
1. Load news headlines and stock price data
2. Perform sentiment analysis on headlines
3. Calculate daily stock returns
4. Determine correlation between sentiment and returns
5. Generate visualizations in the output directory

Detailed instructions for running other analyses are provided in the notebooks directory.

## Contributors

Sentayhu Berhanu

## License
