"""
Main FastAPI application for HAI3 Serving
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import config
from api.routes import router
from models.model_manager import model_manager
from schemas.openai_schemas import ErrorResponse, ErrorDetail

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.server.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting HAI3 Serving Application")
    logger.info(f"Loading model: {config.model.model_name}")
    
    try:
        await model_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down HAI3 Serving Application")


# Create FastAPI application
app = FastAPI(
    title="HAI3 Serving API",
    description="OpenAI-compatible API for HAI3.1 model serving using Transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": None
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HAI3 Serving API",
        "version": "1.0.0",
        "model": config.model.model_name,
        "model_loaded": model_manager.is_loaded()
    }


@app.get("/v1")
async def v1_root():
    """OpenAI v1 API root"""
    return {
        "message": "HAI3 Serving API v1",
        "endpoints": [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions"
        ]
    }


def main():
    """Main entry point"""
    logger.info(f"Starting server on {config.server.host}:{config.server.port}")
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level=config.server.log_level,
        reload=False
    )


if __name__ == "__main__":
    main()