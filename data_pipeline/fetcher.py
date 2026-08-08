import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Popular Nifty 50 tickers with human-readable labels
NIFTY50_TICKERS: Dict[str, str] = {
    "^NSEI": "NIFTY 50 Index",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "TATAMOTORS.NS": "Tata Motors",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen & Toubro",
    "SBIN.NS": "State Bank of India",
    "WIPRO.NS": "Wipro Limited",
    "HCLTECH.NS": "HCL Technologies",
    "AXISBANK.NS": "Axis Bank",
    "SUNPHARMA.NS": "Sun Pharma"
}

def fetch_stock_data(ticker: str = "^NSEI", period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches real-time and historical stock data via yfinance API.
    
    Parameters:
        ticker (str): Yahoo Finance ticker symbol (e.g., '^NSEI', 'RELIANCE.NS')
        period (str): Valid period - 1y, 2y, 5y, max (default: '5y')
        interval (str): Data interval - 1d, 1wk (default: '1d')
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with Date index and Open, High, Low, Close, Volume features.
    """
    logger.info(f"Fetching data for ticker: {ticker} (Period: {period}, Interval: {interval})")
    
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        
        if data.empty:
            raise ValueError(f"No data retrieved for ticker symbol: {ticker}")
            
        # Clean multi-index columns if yfinance returns tuple column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data = data.dropna()
        data.reset_index(inplace=True)
        
        # Format Date column cleanly
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Engineer basic technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Fill missing values from rolling window calculation
        data = data.bfill().ffill()
        
        logger.info(f"Successfully fetched {len(data)} rows of market data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise e

def get_available_tickers() -> Dict[str, str]:
    """Returns dictionary of supported Nifty 50 ticker symbols and names."""
    return NIFTY50_TICKERS

if __name__ == "__main__":
    df = fetch_stock_data("^NSEI", period="1y")
    print(df.head())
    print("Available tickers count:", len(get_available_tickers()))
