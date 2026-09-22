import sqlite3
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
from config import DB_PATH, SCHEMA_PATH

class DatabaseManager:
    """Provides thread-safe connections and CRUD operation helpers for SQLite data warehouse."""
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a raw SQLite connection context."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self, schema_file: str = str(SCHEMA_PATH)) -> None:
        """Executes DDL statements to set up SQLite schema tables and indices."""
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
        print(f"[INFO] Database initialized successfully at: {self.db_path}")

    def execute_query(self, query: str, params: Union[Tuple, List, Dict] = ()) -> pd.DataFrame:
        """Runs a SELECT query and safely converts results into a pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def execute_non_query(self, query: str, params: Union[Tuple, List, Dict] = ()) -> int:
        """Executes INSERT, UPDATE, or DELETE queries within an explicit transaction context."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def execute_batch(self, query: str, param_list: List[Union[Tuple, Dict]]) -> int:
        """Executes optimized batch inserts using executemany."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, param_list)
            conn.commit()
            return cursor.rowcount

if __name__ == "__main__":
    db = DatabaseManager()
    db.initialize_schema()