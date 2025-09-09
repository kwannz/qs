import pandas as pd
from datetime import datetime
from typing import Optional, Union, List, Dict, Any

from panda_common.handlers.database_handler import DatabaseHandler
from panda_common.logger_config import logger


class MarketCryptoReader:
    """
    Minimal crypto market data reader.

    - Reads daily OHLCV-like data from Mongo collection "crypto_market".
    - Inclusive date range with YYYYMMDD strings; optional symbol filter.
    - Fields default to all if not specified.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_handler = DatabaseHandler(config)

    def get_market_data(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[Union[str, List[str]]] = None,
    ) -> Optional[pd.DataFrame]:
        if start_date is None or end_date is None:
            logger.error("start_date and end_date must be provided for crypto market data")
            return None

        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(fields, str):
            fields = [fields]

        query: Dict[str, Any] = {
            "date": {
                "$gte": str(start_date),
                "$lte": str(end_date),
            }
        }
        if symbols:
            query["symbol"] = {"$in": symbols}

        projection = None
        if fields:
            projection = {field: 1 for field in fields + ['date', 'symbol']}
            projection['_id'] = 0

        collection = self.db_handler.get_mongo_collection(
            self.config["MONGO_DB"],
            "crypto_market",
        )

        cursor = collection.find(query, projection=projection)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            logger.warning("No crypto market data found for the specified parameters")
            return None
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        return df

