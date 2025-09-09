from typing import Optional

from panda_server.config.database import mongodb
from datetime import datetime

from panda_trading.models.trading.trading_real_binding import RealTradeBindingModel

REAL_TRADE_BINDING_COLLECTION = "real_trade_binding"


async def find_strategy_account(strategy_id: int, account: str, account_type: int) -> Optional[RealTradeBindingModel]:
    collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)

    document = await collection.find_one({
        "strategy_id": strategy_id,
        "future_account": account,
        "account_type": account_type
    })

    if not document:
        return None

    return RealTradeBindingModel(**document)