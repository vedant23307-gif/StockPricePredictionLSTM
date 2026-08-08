import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates LSTM stock predictions across quantitative accuracy metrics (RMSE, MAE, R², Directional Accuracy).
    Generates next-day forecasts and buy/sell trading signals.
    """
    @staticmethod
    def evaluate(y_true_scaled: np.ndarray, 
                 y_pred_scaled: np.ndarray, 
                 y_true_actual: np.ndarray, 
                 y_pred_actual: np.ndarray) -> Dict[str, float]:
        """
        Calculates comprehensive statistical metrics on scaled and original price data.
        
        Returns:
            Dict containing scaled_rmse (~0.029 target), actual_rmse, mae, mape, r2_score, directional_accuracy.
        """
        # Ensure 1D arrays
        y_true_s = y_true_scaled.flatten()
        y_pred_s = y_pred_scaled.flatten()
        y_true_a = y_true_actual.flatten()
        y_pred_a = y_pred_actual.flatten()

        # Scaled RMSE
        scaled_rmse = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))

        # Actual Price metrics
        actual_rmse = float(np.sqrt(mean_squared_error(y_true_a, y_pred_a)))
        mae = float(mean_absolute_error(y_true_a, y_pred_a))
        
        # Avoid division by zero in MAPE
        non_zero_idx = y_true_a != 0
        mape = float(np.mean(np.abs((y_true_a[non_zero_idx] - y_pred_a[non_zero_idx]) / y_true_a[non_zero_idx])) * 100)
        
        # R2 score
        r2 = float(r2_score(y_true_a, y_pred_a))

        # Directional Accuracy (% correct up/down direction predictions)
        if len(y_true_a) > 1:
            actual_direction = np.diff(y_true_a) > 0
            pred_direction = np.diff(y_pred_a) > 0
            directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
        else:
            directional_accuracy = 0.0

        metrics = {
            "scaled_rmse": round(scaled_rmse, 4),
            "actual_rmse": round(actual_rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "r2_score": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 2)
        }

        logger.info(f"Model Evaluation Metrics Summary:")
        logger.info(f" -> Scaled RMSE:         {metrics['scaled_rmse']} (Target ~ 0.029)")
        logger.info(f" -> Actual Price RMSE:    ₹{metrics['actual_rmse']}")
        logger.info(f" -> MAE:                  ₹{metrics['mae']}")
        logger.info(f" -> MAPE:                 {metrics['mape']}%")
        logger.info(f" -> R² Score:             {metrics['r2_score']}")
        logger.info(f" -> Directional Accuracy: {metrics['directional_accuracy']}%")

        return metrics

    @staticmethod
    def generate_next_day_forecast(last_window_scaled: np.ndarray, 
                                   model_predict_fn, 
                                   scaler, 
                                   last_known_actual_price: float) -> Dict[str, Any]:
        """
        Generates 1-day ahead forecasted stock price and trading signal.
        """
        # Ensure shape (1, 60, 1)
        input_seq = last_window_scaled.reshape(1, last_window_scaled.shape[0], 1)
        next_day_pred_scaled = model_predict_fn(input_seq)
        
        next_day_price = float(scaler.inverse_transform(next_day_pred_scaled)[0][0])
        price_change = next_day_price - last_known_actual_price
        pct_change = (price_change / last_known_actual_price) * 100

        # Signal logic
        if pct_change > 0.5:
            signal = "BUY (BULLISH)"
            confidence = "HIGH" if pct_change > 1.5 else "MODERATE"
        elif pct_change < -0.5:
            signal = "SELL (BEARISH)"
            confidence = "HIGH" if pct_change < -1.5 else "MODERATE"
        else:
            signal = "HOLD (NEUTRAL)"
            confidence = "NEUTRAL"

        return {
            "last_actual_price": round(last_known_actual_price, 2),
            "forecasted_next_price": round(next_day_price, 2),
            "expected_change_amount": round(price_change, 2),
            "expected_change_percent": round(pct_change, 2),
            "trading_signal": signal,
            "signal_confidence": confidence
        }

if __name__ == "__main__":
    y_true_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y_pred_s = np.array([0.11, 0.19, 0.31, 0.39, 0.49])
    res = ModelEvaluator.evaluate(y_true_s, y_pred_s, y_true_s * 100, y_pred_s * 100)
    print(res)
