import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Handles feature scaling (MinMaxScaler) and 60-day window sequence engineering for LSTM models.
    """
    def __init__(self, window_size: int = 60, feature_col: str = "Close", train_split: float = 0.8):
        self.window_size = window_size
        self.feature_col = feature_col
        self.train_split = train_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocesses stock price DataFrame and engineers 60-day sequence windows.
        
        Parameters:
            df (pd.DataFrame): Stock price historical data containing feature_col.
            
        Returns:
            dict containing X_train, y_train, X_test, y_test, scaled_data, dates, and scaler object.
        """
        if self.feature_col not in df.columns:
            raise KeyError(f"Feature column '{self.feature_col}' not found in DataFrame.")

        raw_values = df[[self.feature_col]].values
        
        # Fit scaler on full data (or train subset to prevent data leakage)
        train_size = int(len(raw_values) * self.train_split)
        
        # Fit scaler on training portion only
        self.scaler.fit(raw_values[:train_size])
        self.is_fitted = True
        
        # Transform entire series
        scaled_data = self.scaler.transform(raw_values)

        # Create sliding 60-day sequences
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size : i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        
        # Reshape X for LSTM input: (samples, time_steps, features) -> (N, 60, 1)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split into chronologically ordered Train and Test sets
        split_idx = train_size - self.window_size
        if split_idx <= 0:
            raise ValueError(f"Insufficient data length ({len(df)}) for window_size {self.window_size}")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        dates = df['Date'].iloc[self.window_size:].tolist() if 'Date' in df.columns else list(range(len(y)))

        logger.info(f"Engineered 60-day window sequences successfully:")
        logger.info(f" -> Train Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f" -> Test Shapes:  X_test={X_test.shape}, y_test={y_test.shape}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "scaled_data": scaled_data,
            "raw_data": raw_values,
            "dates": dates,
            "train_size": train_size,
            "scaler": self.scaler,
            "window_size": self.window_size
        }

    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse transforms scaled predictions back to original stock price currency values."""
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        return self.scaler.inverse_transform(scaled_values)

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "Date": pd.date_range(start="2020-01-01", periods=300),
        "Close": np.random.randn(300).cumsum() + 100
    })
    prep = TimeSeriesPreprocessor(window_size=60)
    res = prep.fit_transform(sample_df)
    print("X_train shape:", res["X_train"].shape)
