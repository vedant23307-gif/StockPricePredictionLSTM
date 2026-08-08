import os
import sys
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# Add parent directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data_pipeline.fetcher import fetch_stock_data, get_available_tickers
from data_pipeline.database import DatabaseManager
from data_pipeline.preprocessor import TimeSeriesPreprocessor
from models.lstm_model import StockLSTMModel
from models.evaluator import ModelEvaluator

os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".matplotlib")

app = FastAPI(
    title="Nifty 50 Stock Price Prediction API",
    description="FastAPI Production Engine for 3-Layer LSTM Deep Learning Stock Forecasting",
    version="1.0.0"
)

# Mount Static Files and Templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPIStockApp")

class PredictRequest(BaseModel):
    ticker: str = "^NSEI"
    epochs: int = 20
    force_retrain: bool = False

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Render main glassmorphism stock prediction dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/tickers")
async def get_tickers():
    """Return dictionary of supported Nifty 50 ticker symbols."""
    return {
        "status": "success",
        "tickers": get_available_tickers()
    }

@app.post("/api/predict")
async def predict_stock_post(payload: PredictRequest):
    """Run model training and prediction pipeline via POST payload."""
    return await run_prediction_pipeline(ticker=payload.ticker, epochs=payload.epochs, force_retrain=payload.force_retrain)

@app.get("/api/predict")
async def predict_stock_get(ticker: str = "^NSEI", epochs: int = 20, force_retrain: bool = False):
    """Run model training and prediction pipeline via GET parameters."""
    return await run_prediction_pipeline(ticker=ticker, epochs=epochs, force_retrain=force_retrain)

async def run_prediction_pipeline(ticker: str, epochs: int, force_retrain: bool = False):
    """Executes prediction pipeline and builds API response dictionary."""
    try:
        logger.info(f"FastAPI Prediction requested: Ticker={ticker}, Epochs={epochs}, ForceRetrain={force_retrain}")

        # Step 1: Fetch Market Data via yfinance
        df = fetch_stock_data(ticker=ticker, period="5y")

        # Step 2: Store Data in DB
        db = DatabaseManager()
        db.save_stock_data(df, ticker)

        # Step 3: Preprocess Sequences
        preprocessor = TimeSeriesPreprocessor(window_size=60, train_split=0.8)
        prep_data = preprocessor.fit_transform(df)

        X_train, y_train = prep_data["X_train"], prep_data["y_train"]
        X_test, y_test = prep_data["X_test"], prep_data["y_test"]
        scaler = prep_data["scaler"]

        # Step 4: Model Load or Train
        clean_name = ticker.replace("^", "").replace(".", "_")
        model_file = os.path.join(BASE_DIR, "saved_models", f"lstm_{clean_name}.keras")

        lstm = StockLSTMModel(input_shape=(60, 1))

        if os.path.exists(model_file) and not force_retrain:
            try:
                lstm = StockLSTMModel.load(model_file)
                logger.info(f"Loaded existing pre-trained model for {ticker} instantly")
            except Exception as load_err:
                logger.warning(f"Failed loading cached model ({load_err}), training new model...")
                lstm.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32)
                lstm.save(model_file)
        else:
            logger.info(f"Training new 3-Layer LSTM model for {ticker} ({epochs} epochs)...")
            lstm.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32)
            lstm.save(model_file)

        # Step 5: Generate Predictions
        y_pred_scaled = lstm.predict(X_test)
        y_test_actual = preprocessor.inverse_transform(y_test).flatten().tolist()
        y_pred_actual = preprocessor.inverse_transform(y_pred_scaled).flatten().tolist()

        dates = prep_data["dates"][prep_data["train_size"] - 60:]
        test_dates = [str(d)[:10] for d in dates[:len(y_test_actual)]]

        # Step 6: Evaluate Metrics
        metrics = ModelEvaluator.evaluate(
            y_true_scaled=y_test,
            y_pred_scaled=y_pred_scaled,
            y_true_actual=np.array(y_test_actual),
            y_pred_actual=np.array(y_pred_actual)
        )

        # Step 7: Next-Day Forecast
        last_60_scaled = prep_data["scaled_data"][-60:]
        last_price = float(df["Close"].iloc[-1])
        forecast = ModelEvaluator.generate_next_day_forecast(
            last_window_scaled=last_60_scaled,
            model_predict_fn=lstm.predict,
            scaler=scaler,
            last_known_actual_price=last_price
        )

        return JSONResponse(content={
            "status": "success",
            "ticker": ticker,
            "company_name": get_available_tickers().get(ticker, ticker),
            "dates": test_dates,
            "actual_prices": [round(val, 2) for val in y_test_actual],
            "predicted_prices": [round(val, 2) for val in y_pred_actual],
            "metrics": metrics,
            "forecast": forecast,
            "db_type": db.db_type
        })

    except Exception as e:
        logger.error(f"Error in FastAPI run_prediction_pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
