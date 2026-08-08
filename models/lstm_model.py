import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Ensure KERAS / TF runtime path safety
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class StockLSTMModel:
    """
    3-Layer LSTM Neural Network for Stock Price Prediction built with TensorFlow / Keras.
    Includes Dropout regularization and dynamic learning rate callbacks.
    """
    def __init__(self, input_shape: Tuple[int, int] = (60, 1), units: int = 50, dropout_rate: float = 0.2):
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.model: Optional[Sequential] = None
        
        self.build_model()

    def build_model(self) -> Sequential:
        """
        Constructs a 3-layer LSTM architecture with Dropout regularization layers.
        """
        logger.info(f"Building 3-layer LSTM neural network architecture with input shape: {self.input_shape}")
        
        model = Sequential([
            # Layer 1: LSTM with return_sequences=True
            LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape, name="lstm_layer_1"),
            Dropout(self.dropout_rate, name="dropout_1"),
            
            # Layer 2: LSTM with return_sequences=True
            LSTM(units=self.units, return_sequences=True, name="lstm_layer_2"),
            Dropout(self.dropout_rate, name="dropout_2"),
            
            # Layer 3: LSTM with return_sequences=False
            LSTM(units=self.units, return_sequences=False, name="lstm_layer_3"),
            Dropout(self.dropout_rate, name="dropout_3"),
            
            # FC Layers
            Dense(units=25, activation="relu", name="dense_intermediate"),
            Dense(units=1, name="price_prediction_output")
        ])

        # Compile model using Adam optimizer and Mean Squared Error loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae", "mse"])
        
        self.model = model
        logger.info("3-Layer LSTM model built and compiled successfully.")
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 40, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Trains the 3-Layer LSTM model with EarlyStopping and Learning Rate Reduction callbacks.
        """
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
        ]

        logger.info(f"Starting LSTM model training for {epochs} epochs (Batch size: {batch_size})...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("LSTM model training completed successfully.")
        return history

    def predict(self, X: tf.Tensor or list or tuple or Any) -> Any:
        """Generates predictions for input sequence X."""
        if self.model is None:
            raise RuntimeError("Model is not initialized or trained.")
        return self.model.predict(X, verbose=0)

    def save(self, filepath: str):
        """Saves trained model to disk in .keras format."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to file: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "StockLSTMModel":
        """Loads a pre-trained Keras model from disk."""
        instance = cls()
        instance.model = load_model(filepath)
        logger.info(f"Loaded trained LSTM model from: {filepath}")
        return instance

if __name__ == "__main__":
    import numpy as np
    dummy_X = np.random.rand(100, 60, 1)
    dummy_y = np.random.rand(100, 1)
    lstm = StockLSTMModel(input_shape=(60, 1))
    lstm.model.summary()
