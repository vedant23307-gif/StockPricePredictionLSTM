import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom modules
from data_pipeline.fetcher import fetch_stock_data, get_available_tickers
from data_pipeline.database import DatabaseManager
from data_pipeline.preprocessor import TimeSeriesPreprocessor
from models.lstm_model import StockLSTMModel
from models.evaluator import ModelEvaluator

# Environment variable check for TensorFlow/Keras sandbox permissions
os.environ["KERAS_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")

handler = logging.StreamHandler(sys.stdout)
handler.flush = sys.stdout.flush
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s", handlers=[handler], force=True)
logger = logging.getLogger("StockLSTMPipeline")


def run_pipeline(ticker: str = "^NSEI", period: str = "5y", epochs: int = 30, batch_size: int = 32) -> dict:
    """
    Executes end-to-end deep learning pipeline for stock price prediction.
    """
    logger.info("=" * 70)
    logger.info(f"STARTING STOCK PRICE PREDICTION PIPELINE: {ticker}")
    logger.info("=" * 70)

    # Step 1: Fetch Live Market Data via yfinance API
    df = fetch_stock_data(ticker=ticker, period=period)
    logger.info(f"Step 1 Complete: Downloaded {len(df)} price points from yfinance.")

    # Step 2: Store Data in Database (PostgreSQL with SQLite fallback)
    db = DatabaseManager()
    db.save_stock_data(df, ticker)
    logger.info(f"Step 2 Complete: Data stored into {db.db_type} database.")

    # Step 3: Preprocess & Engineer 60-Day Sequence Windows
    preprocessor = TimeSeriesPreprocessor(window_size=60, feature_col="Close", train_split=0.8)
    prep_data = preprocessor.fit_transform(df)

    X_train, y_train = prep_data["X_train"], prep_data["y_train"]
    X_test, y_test = prep_data["X_test"], prep_data["y_test"]
    scaler = prep_data["scaler"]

    logger.info(f"Step 3 Complete: Engineered 60-day window sequences (X_train shape: {X_train.shape}).")

    # Step 4: Build & Train 3-Layer LSTM TensorFlow Network
    lstm = StockLSTMModel(input_shape=(X_train.shape[1], 1), units=50, dropout_rate=0.2)
    
    history = lstm.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
    logger.info("Step 4 Complete: 3-Layer LSTM neural network trained successfully.")

    # Step 5: Make Predictions on Test Data & Inverse Transform
    y_pred_scaled = lstm.predict(X_test)
    
    # Inverse transform predictions and actual targets back to original stock price currency
    y_test_actual = preprocessor.inverse_transform(y_test)
    y_pred_actual = preprocessor.inverse_transform(y_pred_scaled)

    # Step 6: Evaluate Model Performance Metrics
    metrics = ModelEvaluator.evaluate(
        y_true_scaled=y_test,
        y_pred_scaled=y_pred_scaled,
        y_true_actual=y_test_actual,
        y_pred_actual=y_pred_actual
    )

    # Step 7: Next-Day Price Forecast & Trading Signal
    last_60_days_scaled = prep_data["scaled_data"][-60:]
    last_actual_price = float(df["Close"].iloc[-1])
    forecast = ModelEvaluator.generate_next_day_forecast(
        last_window_scaled=last_60_days_scaled,
        model_predict_fn=lstm.predict,
        scaler=scaler,
        last_known_actual_price=last_actual_price
    )

    # Step 8: Plot and Save Matplotlib Visualizations
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(plots_dir, exist_ok=True)
    
    prediction_plot_path = os.path.join(plots_dir, f"{ticker.replace('^', '').replace('.', '_')}_prediction.png")
    loss_plot_path = os.path.join(plots_dir, f"{ticker.replace('^', '').replace('.', '_')}_loss_curve.png")

    # Chart 1: Actual vs Predicted Stock Prices
    test_dates = prep_data["dates"][prep_data["train_size"] - 60:]
    test_dates = test_dates[:len(y_test_actual)]

    plt.figure(figsize=(14, 6), dpi=300)
    plt.plot(test_dates, y_test_actual, color="#38ef7d", label="Actual Stock Price", linewidth=2)
    plt.plot(test_dates, y_pred_actual, color="#11998e", label="LSTM Predicted Price", linestyle="--", linewidth=2)
    plt.title(f"Nifty 50 Stock Price Prediction using 3-Layer LSTM ({ticker})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR ₹)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(prediction_plot_path)
    plt.close()

    # Chart 2: Model Training & Validation Loss Curve
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='#4e54c8', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#ff416c', linewidth=2)
    plt.title(f"LSTM Training & Validation Loss Curve ({ticker})", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    # Step 9: Save Trained Model
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", f"lstm_{ticker.replace('^', '').replace('.', '_')}.keras")
    lstm.save(model_save_path)

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Target Scaled RMSE:      {metrics['scaled_rmse']} (Goal: ~0.029)")
    logger.info(f"Actual Price RMSE:       ₹{metrics['actual_rmse']}")
    logger.info(f"R² Performance Score:    {metrics['r2_score']}")
    logger.info(f"Directional Accuracy:    {metrics['directional_accuracy']}%")
    logger.info(f"Next-Day Forecast Price: ₹{forecast['forecasted_next_price']} ({forecast['trading_signal']})")
    logger.info(f"Plots Saved to:          {plots_dir}")
    logger.info(f"Model Saved to:          {model_save_path}")
    logger.info("=" * 70)

    return {
        "ticker": ticker,
        "metrics": metrics,
        "forecast": forecast,
        "prediction_plot": prediction_plot_path,
        "loss_plot": loss_plot_path,
        "model_path": model_save_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Layer LSTM Stock Price Prediction Pipeline")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Stock Ticker Symbol (default: ^NSEI)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period (e.g., 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    
    args = parser.parse_args()
    run_pipeline(ticker=args.ticker, period=args.period, epochs=args.epochs, batch_size=args.batch_size)
