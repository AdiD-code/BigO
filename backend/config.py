# backend/config.py

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if it exists
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/tufanticket")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "tufanticket")

# API Keys
MERAEVENTS_API_KEY = os.getenv("MERAEVENTS_API_KEY", "")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY", "eGQ7RX5AeS2okEA2s9Y3sBgWG2GEA5ZE")
BOOKMYSHOW_SCRAPER_ENABLED = os.getenv("BOOKMYSHOW_SCRAPER_ENABLED", "False").lower() == "true"

# Kaggle Configuration
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Cache Settings
CACHE_EXPIRY_SECONDS = int(os.getenv("CACHE_EXPIRY_SECONDS", "3600"))  # Default: 1 hour
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))  # Max number of items in cache

# Data Refresh Settings
AUTO_REFRESH_INTERVAL_HOURS = int(os.getenv("AUTO_REFRESH_INTERVAL_HOURS", "1"))  # Default: daily
DEFAULT_CITIES = os.getenv("DEFAULT_CITIES", "Mumbai,Delhi,Bangalore,Hyderabad,Chennai").split(",")

# API Settings
API_VERSION = os.getenv("API_VERSION", "v1")
API_PREFIX = f"/api/{API_VERSION}"
API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "*").split(",")
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))  # Requests per hour
API_RATE_LIMIT_WINDOW = int(os.getenv("API_RATE_LIMIT_WINDOW", "3600"))  # Window in seconds

# Recommendation Engine Settings
RECOMMENDATION_ALGORITHM = os.getenv("RECOMMENDATION_ALGORITHM", "collaborative")  # Options: collaborative, content, hybrid
RECOMMENDATION_MIN_INTERACTIONS = int(os.getenv("RECOMMENDATION_MIN_INTERACTIONS", "5"))
SIMILAR_EVENTS_COUNT = int(os.getenv("SIMILAR_EVENTS_COUNT", "10"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = os.getenv("LOG_FILE", "logs/tufanticket.log")
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "True").lower() == "true"

# Set up numeric log levels
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

NUMERIC_LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

# Application Settings
APP_NAME = os.getenv("APP_NAME", "TufanTicket")
APP_ENV = os.getenv("APP_ENV", "development")  # Options: development, testing, production
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true" and APP_ENV != "production"

# User Settings
USER_PASSWORD_MIN_LENGTH = int(os.getenv("USER_PASSWORD_MIN_LENGTH", "8"))
USER_SESSION_EXPIRY_DAYS = int(os.getenv("USER_SESSION_EXPIRY_DAYS", "30"))
USER_VERIFICATION_REQUIRED = os.getenv("USER_VERIFICATION_REQUIRED", "False").lower() == "true"

# Email Settings (for notifications, verification, etc.)
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"
EMAIL_SENDER = os.getenv("EMAIL_SENDER", f"noreply@{APP_NAME.lower()}.com")
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.example.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER", "")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() == "true"

# Feature Flags
FEATURE_SOCIAL_LOGIN = os.getenv("FEATURE_SOCIAL_LOGIN", "True").lower() == "true"
FEATURE_EVENT_REVIEWS = os.getenv("FEATURE_EVENT_REVIEWS", "True").lower() == "true"
FEATURE_ORGANIZER_ANALYTICS = os.getenv("FEATURE_ORGANIZER_ANALYTICS", "True").lower() == "true"
FEATURE_ADVANCED_RECOMMENDATIONS = os.getenv("FEATURE_ADVANCED_RECOMMENDATIONS", "True").lower() == "true"

# Default values for missing data
DEFAULT_EVENT_IMAGE = os.getenv("DEFAULT_EVENT_IMAGE", "https://via.placeholder.com/500x300?text=Event")
DEFAULT_USER_AVATAR = os.getenv("DEFAULT_USER_AVATAR", "https://via.placeholder.com/150?text=User")

# Development-only settings
if APP_ENV == "development":
    # For local development without real API keys
    USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "True").lower() == "true"
    MOCK_DATA_SIZE = int(os.getenv("MOCK_DATA_SIZE", "100"))
else:
    USE_MOCK_DATA = False
    MOCK_DATA_SIZE = 0

# Testing settings
if APP_ENV == "testing":
    # Override MongoDB connection for testing
    MONGODB_URI = os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017/tufanticket_test")
    MONGODB_DB_NAME = os.getenv("TEST_MONGODB_DB_NAME", "tufanticket_test")