"""
data/extractor.py

Handles all database connectivity and data extraction using SQLAlchemy,
with SQL queries stored as separate .sql files for maintainability.

SQLAlchemy is preferred over raw pyodbc here because:
    - pd.read_sql() works natively with SQLAlchemy engines
    - Named parameters via text() are cleaner and safer than ? placeholders
    - Engine connection pooling is handled automatically

SQL files live in data/sql/ and are loaded via _read_sql_file().
This keeps query logic separate from Python, making it easy to edit,
version-control, and review SQL independently.

Usage
-----
    # Basic usage
    with DataExtractor(connection_string) as db:
        df = db.get_combined_data('2021-01-01', '2022-12-31')

    # Or with a custom SQL file
    with DataExtractor(connection_string) as db:
        df = db.query(
            Path("data/sql/my_custom_query.sql"),
            params={"start_date": "2021-01-01", "end_date": "2022-12-31"}
        )

Connection string examples
--------------------------
    SQL Server (Windows auth):
        "mssql+pyodbc://server/database?driver=ODBC+Driver+17+for+SQL+Server"

    SQL Server (SQL auth):
        "mssql+pyodbc://user:password@server/database
         ?driver=ODBC+Driver+17+for+SQL+Server"

    PostgreSQL:
        "postgresql+psycopg2://user:password@host/database"
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Dict, Optional, Union


# Default location of SQL files, relative to this file
_SQL_DIR = Path(__file__).parent / "sql"


class DataExtractor:
    """
    Manages SQLAlchemy engine connections and SQL-file-based data extraction.

    Implements context manager so connections are always closed cleanly,
    even if an error occurs mid-query.
    """

    def __init__(self, connection_string: str):
        """
        Parameters
        ----------
        connection_string : str
            SQLAlchemy connection string for the target database.

            SQL Server (Windows auth):
                "mssql+pyodbc://server/database
                 ?driver=ODBC+Driver+17+for+SQL+Server"

            SQL Server (SQL auth):
                "mssql+pyodbc://user:password@server/database
                 ?driver=ODBC+Driver+17+for+SQL+Server"
        """
        self.connection_string = connection_string
        self._engine: Optional[Engine] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "DataExtractor":
        self._engine = create_engine(self.connection_string)
        return self

    def disconnect(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "DataExtractor":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # SQL file reader
    # ------------------------------------------------------------------

    @staticmethod
    def _read_sql_file(path: Union[str, Path]) -> str:
        """
        Read a .sql file and return its contents as a string.

        Parameters
        ----------
        path : path to the .sql file, either absolute or relative to cwd.

        Raises
        ------
        FileNotFoundError if the .sql file does not exist at the given path.
        """
        sql_path = Path(path)
        if not sql_path.exists():
            raise FileNotFoundError(
                f"SQL file not found: {sql_path.resolve()}\n"
                f"Check the path relative to your working directory."
            )
        with open(sql_path, "r") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def query(
        self,
        sql_path: Union[str, Path],
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Read a .sql file and execute it, returning results as a DataFrame.

        Uses SQLAlchemy's text() so named parameters (e.g. :start_date)
        in the SQL file are safely bound before execution. This avoids
        string interpolation and protects against SQL injection.

        Parameters
        ----------
        sql_path : path to the .sql file to execute
        params   : dict of named parameters matching :param_name
                   placeholders in the SQL file.

                   Example:
                       params={"start_date": "2021-01-01",
                               "end_date":   "2022-12-31"}

        Returns
        -------
        pd.DataFrame of query results.

        Example
        -------
        df = db.query(
            "data/sql/combined_data.sql",
            params={"start_date": "2021-01-01", "end_date": "2022-12-31"}
        )
        """
        if self._engine is None:
            raise RuntimeError(
                "No active connection. Use connect() or a 'with' block."
            )

        raw_sql   = self._read_sql_file(sql_path)
        bound_sql = text(raw_sql)

        with self._engine.connect() as conn:
            return pd.read_sql(bound_sql, conn, params=params or {})

    # ------------------------------------------------------------------
    # Named extraction methods
    # Each points to the corresponding .sql file in data/sql/
    # Adjust the SQL files themselves to match your schema —
    # no Python changes needed.
    # ------------------------------------------------------------------

    def get_customer_data(
        self,
        start_date: str,
        end_date: str,
        sql_path: Union[str, Path] = _SQL_DIR / "customer_data.sql",
    ) -> pd.DataFrame:
        """
        Pull customer-level variables for the PD_cust component.

        Parameters
        ----------
        start_date : inclusive start of application_date range (YYYY-MM-DD)
        end_date   : inclusive end of application_date range (YYYY-MM-DD)
        sql_path   : override to use a custom SQL file if needed
        """
        return self.query(
            sql_path,
            params={"start_date": start_date, "end_date": end_date},
        )

    def get_deal_data(
        self,
        start_date: str,
        end_date: str,
        sql_path: Union[str, Path] = _SQL_DIR / "deal_data.sql",
    ) -> pd.DataFrame:
        """
        Pull deal-level variables for the f(deal) component.

        Parameters
        ----------
        start_date : inclusive start of application_date range (YYYY-MM-DD)
        end_date   : inclusive end of application_date range (YYYY-MM-DD)
        sql_path   : override to use a custom SQL file if needed
        """
        return self.query(
            sql_path,
            params={"start_date": start_date, "end_date": end_date},
        )

    def get_combined_data(
        self,
        start_date: str,
        end_date: str,
        sql_path: Union[str, Path] = _SQL_DIR / "combined_data.sql",
    ) -> pd.DataFrame:
        """
        Pull customer and deal variables in a single joined query.
        This is the primary extraction used by the pipeline.

        Parameters
        ----------
        start_date : inclusive start of application_date range (YYYY-MM-DD)
        end_date   : inclusive end of application_date range (YYYY-MM-DD)
        sql_path   : override to use a custom SQL file if needed
        """
        return self.query(
            sql_path,
            params={"start_date": start_date, "end_date": end_date},
        )

    def get_monitoring_data(
        self,
        start_date: str,
        end_date: str,
        sql_path: Union[str, Path] = _SQL_DIR / "monitoring_data.sql",
    ) -> pd.DataFrame:
        """
        Pull data for ongoing PSI/CSI monitoring, including model scores
        if already stored alongside raw variable values.

        Parameters
        ----------
        start_date : inclusive start of application_date range (YYYY-MM-DD)
        end_date   : inclusive end of application_date range (YYYY-MM-DD)
        sql_path   : override to use a custom SQL file if needed
        """
        return self.query(
            sql_path,
            params={"start_date": start_date, "end_date": end_date},
        )