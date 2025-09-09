import asyncio
from bson import ObjectId
from panda_server.config.database import mongodb
import logging

logger = logging.getLogger(__name__)

async def workflow_demo_migration_script(workflow_id: str, session=None):
    # 1. 更新 workflow.owner
    workflow_collection = mongodb.get_collection("workflow")
    workflow = await workflow_collection.find_one({"_id": ObjectId(workflow_id)}, session=session)
    if not workflow:
        logger.error(f"Workflow not found: {workflow_id}")
        return False
    await workflow_collection.update_one({"_id": ObjectId(workflow_id)}, {"$set": {"owner": "*"}}, session=session)
    logger.info(f"Updated workflow {workflow_id} owner to *")

    # 2. 获取 last_run_id
    last_run_id = workflow.get("last_run_id")
    if not last_run_id:
        logger.warning(f"Workflow {workflow_id} has no last_run_id")
        return True

    # 3. 更新 workflow_run.owner
    workflow_run_collection = mongodb.get_collection("workflow_run")
    workflow_run = await workflow_run_collection.find_one({"_id": ObjectId(last_run_id)}, session=session)
    if not workflow_run:
        logger.warning(f"WorkflowRun not found: {last_run_id}")
        return True
    await workflow_run_collection.update_one({"_id": ObjectId(last_run_id)}, {"$set": {"owner": "*"}}, session=session)
    logger.info(f"Updated workflow_run {last_run_id} owner to *")

    # 4. 获取 output_data_obj
    output_data_obj = workflow_run.get("output_data_obj", {})
    output_db_ids = list(output_data_obj.values())
    if not output_db_ids:
        logger.info(f"No output_db_ids to update for workflow_run {last_run_id}")
        return True

    # 5. 批量更新 workflow_node_output_fs.files 的 metadata.owner
    files_collection = mongodb.get_collection("workflow_node_output_fs.files")
    result = await files_collection.update_many(
        {"_id": {"$in": [ObjectId(fid) for fid in output_db_ids]}},
        {"$set": {"metadata.owner": "*"}},
        session=session
    )
    logger.info(f"Updated {result.modified_count} files' metadata.owner to *")
    return True

async def main():
    await mongodb.connect_db()
    workflow_ids = input("Please enter workflow_id(s) (separated by commas): ").strip()
    workflow_id_list = [wid.strip() for wid in workflow_ids.split(",") if wid.strip()]
    failed_ids = []
    client = mongodb.client
    async with await client.start_session() as session:
        try:
            async with session.start_transaction():
                for workflow_id in workflow_id_list:
                    print(f"Processing workflow_id: {workflow_id}")
                    success = await workflow_demo_migration_script(workflow_id, session=session)
                    if not success:
                        failed_ids.append(workflow_id)
                        raise Exception(f"Migration failed for workflow_id {workflow_id}")
                    else:
                        print(f"Migration for workflow_id {workflow_id} succeeded, continuing...")
        except Exception as e:
            print(f"Transaction aborted. Error: {e}")
            if failed_ids:
                print(f"The following workflow_id(s) failed: {', '.join(failed_ids)}")
            else:
                print("Migration failed due to an unexpected error.")
            await mongodb.close_db()
            return
    print("All migrations completed!")
    await mongodb.close_db()

if __name__ == "__main__":
    asyncio.run(main()) 