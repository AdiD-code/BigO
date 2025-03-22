"""
Main entry point for the TufanTicket API server.
Sets up FastAPI application, middleware, and registers all routes.
"""

import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pymongo import MongoClient
from fastapi.openapi.utils import get_openapi

# Import route modules
from backend.routes import events, users, insights
from backend.config import (
    APP_ENV, 
    API_VERSION, 
    MONGODB_URI, 
    MONGODB_DB_NAME,
    CORS_ORIGINS,
    LOG_LEVEL,
    API_RATE_LIMIT,
    ENABLE_DOCS
)

# Setup logging
logging_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("tufanticket")

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DB_NAME]
    # Ping the database to verify connection
    db.command("ping")
    logger.info(f"Connected to MongoDB: {MONGODB_DB_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise


# Create FastAPI app
app = FastAPI(
    title="TufanTicket API",
    description="AI-powered event discovery and recommendation system",
    version=API_VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None
)


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TufanTicket API",
        version=API_VERSION,
        description="API for TufanTicket: AI-powered event discovery and recommendation system",
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include database in request state
@app.middleware("http")
async def add_db_to_request(request: Request, call_next):
    request.state.db = db
    response = await call_next(request)
    return response


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # In a real implementation, you'd use proper rate limiting
    # Based on client IP, API key, etc.
    # For now, we're just passing through all requests
    response = await call_next(request)
    return response


# Error handling for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )


# Register all router modules
app.include_router(events.router)
app.include_router(users.router)
app.include_router(insights.router)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    try:
        # Check MongoDB connection
        db.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
        logger.warning("Health check: MongoDB connection failed")
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "version": API_VERSION,
        "environment": APP_ENV,
        "database": db_status
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "TufanTicket API",
        "version": API_VERSION,
        "documentation": "/docs" if ENABLE_DOCS else "Documentation is disabled in this environment"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Determine host and port based on environment
    host = "0.0.0.0" if APP_ENV in ["production", "staging"] else "127.0.0.1"
    port = int(os.environ.get("PORT", 8000))
    
    # Configure uvicorn server
    reload_flag = APP_ENV == "development"
    workers = 1 if APP_ENV == "development" else 4
    
    logger.info(f"Starting TufanTicket API server in {APP_ENV} mode")
    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=reload_flag,
        workers=workers,
        log_level=LOG_LEVEL.lower()
    )