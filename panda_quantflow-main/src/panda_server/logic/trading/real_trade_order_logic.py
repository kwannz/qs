from typing import Optional, List
from bson import ObjectId
from datetime import datetime

from panda_server.config.database import mongodb
from panda_trading.models.trading.trading_real_order import FutureTradeOrderModel


FUTURE_TRADE_RECORD_COLLECTION = "future_trade_orders"


async def get_future_trade_record_by_id(record_id: str) -> Optional[FutureTradeOrderModel]:
    collection = mongodb.get_collection(FUTURE_TRADE_RECORD_COLLECTION)
    document = await collection.find_one({"_id": ObjectId(record_id)})
    if not document:
        return None
    return FutureTradeOrderModel(**document)


async def get_all_future_trade_orders() -> List[FutureTradeOrderModel]:
    collection = mongodb.get_collection(FUTURE_TRADE_RECORD_COLLECTION)
    cursor = collection.find({})
    documents = await cursor.to_list(length=None)
    return [FutureTradeOrderModel(**doc) for doc in documents]


async def create_future_trade_record(record: FutureTradeOrderModel) -> FutureTradeOrderModel:
    collection = mongodb.get_collection(FUTURE_TRADE_RECORD_COLLECTION)

    record_dict = record.model_dump(by_alias=True, exclude_none=True)
    result = await collection.insert_one(record_dict)
    inserted_record = await collection.find_one({"_id": result.inserted_id})
    return FutureTradeOrderModel(**inserted_record)


async def update_future_trade_record(
    record_id: str,
    update_data: dict
) -> Optional[FutureTradeOrderModel]:
    collection = mongodb.get_collection(FUTURE_TRADE_RECORD_COLLECTION)

    update_data["update_time"] = datetime.now()
    result = await collection.update_one(
        {"_id": ObjectId(record_id)},
        {"$set": update_data}
    )

    if result.modified_count == 0:
        return None

    updated_record = await collection.find_one({"_id": ObjectId(record_id)})
    return FutureTradeOrderModel(**updated_record)


async def delete_future_trade_record(record_id: str) -> bool:
    collection = mongodb.get_collection(FUTURE_TRADE_RECORD_COLLECTION)
    result = await collection.delete_one({"_id": ObjectId(record_id)})
    return result.deleted_count == 1