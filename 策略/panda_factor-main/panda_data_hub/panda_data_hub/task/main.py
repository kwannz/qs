import time
import argparse
from typing import Optional
from panda_common.logger_config import logger

from .data_scheduler import DataScheduler
from .factor_clean_scheduler import FactorCleanerScheduler


def run_http_server(host: str, port: int):
    try:
        from fastapi import FastAPI
        import uvicorn
    except Exception:
        logger.warning("未安装 fastapi/uvicorn，HTTP 健康检查不可用。")
        return

    app = FastAPI(title="Panda DataHub Schedulers", version="1.0.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    logger.info(f"启动 HTTP 健康检查服务: http://{host}:{port}/health")
    uvicorn.run(app, host=host, port=port)


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Panda Data/Factor Schedulers Runner")
    parser.add_argument("--only", choices=["data", "factor", "both"], default="both", help="仅启动指定调度器")
    parser.add_argument("--run-now", action="store_true", help="启动前立即执行一次清洗/更新任务")
    parser.add_argument("--http", type=str, default=None, help="启用HTTP健康检查，如 0.0.0.0:8787")

    args = parser.parse_args(argv)

    logger.info("启动数据与因子定时任务调度器 ...")

    data_scheduler = DataScheduler() if args.only in ("data", "both") else None
    factor_scheduler = FactorCleanerScheduler() if args.only in ("factor", "both") else None

    # 立即执行一次
    if args.run_now:
        try:
            if data_scheduler:
                data_scheduler._process_data()
            if factor_scheduler:
                factor_scheduler._process_factor()
        except Exception as e:
            logger.error(f"立即执行失败: {e}")

    # 注册定时任务
    if data_scheduler:
        data_scheduler.schedule_data()
    if factor_scheduler:
        factor_scheduler.schedule_data()

    logger.info("定时任务已注册完成，进程将常驻以保持调度器运行。按 Ctrl+C 退出。")

    # 可选启动HTTP健康检查
    if args.http:
        try:
            host, port_str = args.http.split(":", 1)
            port = int(port_str)
            import threading
            t = threading.Thread(target=run_http_server, args=(host, port), daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"启动健康检查失败: {e}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在停止调度器 ...")
        try:
            if data_scheduler:
                data_scheduler.stop()
        except Exception:
            pass
        try:
            if factor_scheduler:
                factor_scheduler.stop()
        except Exception:
            pass
        logger.info("调度器已停止，退出进程。")


if __name__ == "__main__":
    main()
