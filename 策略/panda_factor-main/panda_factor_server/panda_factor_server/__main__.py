import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from panda_factor_server.routes import user_factor_pro
from panda_llm.routes import chat_router
import mimetypes
from pathlib import Path
from starlette.staticfiles import StaticFiles
from panda_common.logger_config import logger

app = FastAPI(
    title="Panda Server",
    description="Server for Panda AI Factor System",
    version="1.0.0"
)

# Configure CORS (read from env CORS_ALLOW_ORIGINS, comma-separated; default *)
cors_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(user_factor.router, prefix="/api/v1", tags=["user_factors"])
app.include_router(user_factor_pro.router, prefix="/api/v1", tags=["user_factors"])
app.include_router(chat_router.router, prefix="/llm", tags=["panda_llm"])

# 获取根目录下的panda_web
frontend_folder = Path(__file__).resolve().parent.parent.parent / "panda_web" / "panda_web" / "static"
logger.info(f"前端静态资源文件夹路径: {frontend_folder}")
# 显式设置 .js .css 的 MIME 类型
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
# Mount the Vue dist directory at /factor path
app.mount("/factor", StaticFiles(directory=frontend_folder, html=True), name="static")

@app.get("/")
async def home():
    return {"message": "Welcome to the Panda Server!"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)

if __name__ == "__main__":
    main()
