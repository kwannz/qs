import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any

from panda_common.handlers.database_handler import DatabaseHandler
from panda_common.logger_config import logger


class MarketCryptoMinReader:
    """
    Minute-level crypto market reader (24/7).

    - Expects Mongo collection "crypto_market_1m" with fields at least
      ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
    - Query by inclusive calendar date range (YYYYMMDD). You can filter by symbol.
    - Returns a DataFrame or None if no data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_handler = DatabaseHandler(config)

    def _day_bounds(self, yyyymmdd: str):
        start_dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        end_dt = start_dt + timedelta(days=1)
        return start_dt, end_dt

    def get_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[Union[str, List[str]]] = None,
    ) -> Optional[pd.DataFrame]:
        if not start_date or not end_date:
            logger.error("start_date and end_date must be provided for crypto minute data")
            return None

        if isinstance(fields, str):
            fields = [fields]

        # Build datetime bounds
        start_dt, _ = self._day_bounds(str(start_date))
        end_dt_end, _ = self._day_bounds(str(end_date))
        # end bound should be next day's 00:00
        end_bound = end_dt_end + timedelta(days=1)

        query: Dict[str, Any] = {
            "datetime": {
                "$gte": start_dt,
                "$lt": end_bound,
            }
        }
        if symbol:
            query["symbol"] = symbol

        projection = {'_id': 0}
        if fields:
            projection.update({f: 1 for f in fields})
            # Ensure symbol and datetime present to identify records
            projection.setdefault('symbol', 1)
            projection.setdefault('datetime', 1)

        collection = self.db_handler.get_mongo_collection(
            self.config["MONGO_DB"],
            "crypto_market_1m",
        )

        logger.debug(f"crypto_min.query: {query} projection={projection}")
        cursor = collection.find(query, projection=projection)
        df = pd.DataFrame(list(cursor))
        logger.debug(f"crypto_min.count: {len(df)}")
        if df.empty:
            logger.warning("No crypto minute data found for the specified parameters")
            return None
        return df
