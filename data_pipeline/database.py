import os
import sqlite3
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from typing import Optional

logger = logging.getLogger(__name__)

# Default Database Configuration
POSTGRES_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/nifty50_db")
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nifty50_stocks.db")
SQLITE_URL = f"sqlite:///{os.path.abspath(SQLITE_DB_PATH)}"

class DatabaseManager:
    """
    Manages data ingestion and retrieval with PostgreSQL support and automatic SQLite fallback.
    """
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or POSTGRES_URL
        self.engine = None
        self.db_type = "PostgreSQL"
        
        self._init_connection()

    def _init_connection(self):
        """Attempts connection to PostgreSQL, falls back gracefully to SQLite if unavailable."""
        try:
            engine = create_engine(self.db_url, connect_args={"connect_timeout": 3})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.engine = engine
            self.db_type = "PostgreSQL"
            logger.info("Successfully connected to PostgreSQL Database.")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL ({e}). Falling back to SQLite DB...")
            self.engine = create_engine(SQLITE_URL)
            self.db_type = "SQLite"
            logger.info(f"Using SQLite database engine at {SQLITE_DB_PATH}")

    def save_stock_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Saves stock data DataFrame into database table named after ticker.
        
        Parameters:
            df (pd.DataFrame): Stock price data DataFrame
            ticker (str): Ticker symbol name
        """
        table_name = self._clean_table_name(ticker)
        try:
            # Ensure Date column is string/datetime formatted
            df_to_save = df.copy()
            if 'Date' in df_to_save.columns:
                df_to_save['Date'] = df_to_save['Date'].astype(str)
                
            df_to_save.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Saved {len(df_to_save)} records into [{self.db_type}] table: '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Error saving data to table '{table_name}': {str(e)}")
            return False

    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Loads stock data DataFrame from database table.
        
        Parameters:
            ticker (str): Ticker symbol name
            
        Returns:
            pd.DataFrame: Cleaned DataFrame loaded from database.
        """
        table_name = self._clean_table_name(ticker)
        try:
            query = f"SELECT * FROM {table_name} ORDER BY Date ASC"
            df = pd.read_sql_query(query, self.engine)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            logger.info(f"Loaded {len(df)} records from [{self.db_type}] table: '{table_name}'")
            return df
        except Exception as e:
            logger.error(f"Error loading table '{table_name}' from database: {str(e)}")
            raise e

    def list_tables(self) -> list:
        """Lists all stock tables stored in the database."""
        try:
            with self.engine.connect() as conn:
                if self.db_type == "PostgreSQL":
                    query = text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                    result = conn.execute(query)
                    return [row[0] for row in result]
                else:
                    query = text("SELECT name FROM sqlite_master WHERE type='table'")
                    result = conn.execute(query)
                    return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error listing database tables: {str(e)}")
            return []

    @staticmethod
    def _clean_table_name(ticker: str) -> str:
        """Formats ticker name into clean SQL identifier."""
        return "stock_" + ticker.replace("^", "IDX_").replace(".", "_").lower()

if __name__ == "__main__":
    db = DatabaseManager()
    print("Database system initialized. DB Type:", db.db_type)
