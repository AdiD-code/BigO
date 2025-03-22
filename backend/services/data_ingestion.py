# backend/services/data_ingestion.py

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
from pymongo import MongoClient
from functools import lru_cache
import time
from backend.utils.logger import get_logger

# Initialize logging
from utils.logger import get_logger
logger = get_logger(__name__)

# Load configuration
from config import (
    MONGODB_URI, 
    MERAEVENTS_API_KEY, 
    BOOKMYSHOW_SCRAPER_ENABLED,
    TICKETMASTER_API_KEY,
    CACHE_EXPIRY_SECONDS,
    KAGGLE_USERNAME,
    KAGGLE_KEY
)

# MongoDB connection
client = MongoClient(MONGODB_URI)
db = client.tufanticket
events_collection = db.events
kaggle_collection = db.kaggle_data

# Cache setup
EVENT_CACHE = {}
CACHE_TIMESTAMP = {}

# =================== API Integration ===================

def fetch_meraevents(city: str, category: Optional[str] = None) -> List[Dict]:
    """
    Fetch events from MeraEvents API with city and optional category filter
    """
    try:
        logger.info(f"Fetching MeraEvents data for {city}, category: {category}")
        base_url = "https://api.meraevents.com/v1/events"
        
        params = {
            "api_key": MERAEVENTS_API_KEY,
            "city": city,
            "limit": 100,
        }
        
        if category:
            params["category"] = category
            
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Process and standardize event format
        standardized_events = []
        for event in data.get("events", []):
            standardized_event = {
                "event_id": f"me_{event.get('id')}",
                "source": "meraevents",
                "title": event.get("name"),
                "description": event.get("description", ""),
                "location": {
                    "city": city,
                    "venue": event.get("venue", {}).get("name", ""),
                    "address": event.get("venue", {}).get("address", ""),
                    "lat": event.get("venue", {}).get("latitude"),
                    "lon": event.get("venue", {}).get("longitude"),
                },
                "datetime": {
                    "start": event.get("start_time"),
                    "end": event.get("end_time"),
                },
                "category": event.get("category", category or "Other"),
                "pricing": {
                    "min": event.get("min_price", 0),
                    "max": event.get("max_price", 0),
                    "currency": "INR"
                },
                "popularity": event.get("views", 0),
                "image_url": event.get("image_url", ""),
                "ticket_url": event.get("ticket_url", ""),
                "last_updated": datetime.now().isoformat()
            }
            standardized_events.append(standardized_event)
        
        return standardized_events
        
    except Exception as e:
        logger.error(f"Error fetching MeraEvents data: {str(e)}")
        return []

def fetch_events_from_meraevent(city: str, category: Optional[str] = None):
    return fetch_meraevents(city, category)

def scrape_bookmyshow(city: str, category: Optional[str] = None) -> List[Dict]:
    """
    Scrape events from BookMyShow for a given city and optional category
    """
    if not BOOKMYSHOW_SCRAPER_ENABLED:
        logger.info("BookMyShow scraper is disabled")
        return []
    
    try:
        logger.info(f"Scraping BookMyShow for {city}, category: {category}")
        
        # BookMyShow city codes mapping
        city_codes = {
            "mumbai": "MUMBAI",
            "delhi": "NCR",
            "bangalore": "BANG",
            "hyderabad": "HYD",
            "chennai": "CHEN",
            "kolkata": "KOLK",
            "pune": "PUNE",
            # Add more cities as needed
        }
        
        city_code = city_codes.get(city.lower(), city.upper())
        
        # Category mapping
        category_mapping = {
            "music": "Music Shows",
            "comedy": "Comedy Shows",
            "workshops": "Workshops",
            "theatre": "Plays",
            "sports": "Sports",
            # Add more categories as needed
        }
        
        bms_category = category_mapping.get(category.lower(), "") if category else ""
        
        # This would be a more complex scraping logic in reality
        # Here we're simulating the response with a request to their API or website
        
        # Note: In a real implementation, this would use BeautifulSoup, Selenium, or similar tools
        # to scrape the actual BookMyShow website, as they don't have a public API
        
        base_url = f"https://in.bookmyshow.com/explore/events-{city_code}"
        
        # This is a placeholder - actual implementation would parse HTML
        events = [
            {
                "event_id": f"bms_{i}_{int(time.time())}",
                "source": "bookmyshow",
                "title": f"{bms_category or 'Event'} in {city} #{i}",
                "description": f"This is a simulated BookMyShow event #{i} for {city}",
                "location": {
                    "city": city,
                    "venue": f"Venue #{i}",
                    "address": f"Address #{i}, {city}",
                    "lat": None,
                    "lon": None
                },
                "datetime": {
                    "start": (datetime.now() + timedelta(days=i)).isoformat(),
                    "end": (datetime.now() + timedelta(days=i, hours=3)).isoformat()
                },
                "category": bms_category or category or "Other",
                "pricing": {
                    "min": 500 * i,
                    "max": 2000 * i,
                    "currency": "INR"
                },
                "popularity": i * 100,
                "image_url": f"https://example.com/image_{i}.jpg",
                "ticket_url": f"https://in.bookmyshow.com/event/dummy-{i}",
                "last_updated": datetime.now().isoformat()
            }
            for i in range(1, 11)  # Simulate 10 events
        ]
        
        return events
        
    except Exception as e:
        logger.error(f"Error scraping BookMyShow: {str(e)}")
        return []

def fetch_ticketmaster(city: str, category: Optional[str] = None) -> List[Dict]:
    """
    Fetch events from Ticketmaster API (global source)
    """
    if not TICKETMASTER_API_KEY:
        logger.info("Ticketmaster API key not provided, skipping")
        return []
    
    try:
        logger.info(f"Fetching Ticketmaster data for {city}, category: {category}")
        
        # Map Indian cities to country code and city
        # (Ticketmaster needs country code and uses different city names sometimes)
        city_mapping = {
            "mumbai": "Mumbai",
            "delhi": "New Delhi",
            "bangalore": "Bengaluru",
            "hyderabad": "Hyderabad",
            "chennai": "Chennai",
            # Add other cities as needed
        }
        
        # Map our categories to Ticketmaster's segmentId or classificationName
        category_mapping = {
            "music": "KZFzniwnSyZfZ7v7nJ",  # Music segmentId
            "sports": "KZFzniwnSyZfZ7v7nE",  # Sports segmentId
            "arts": "KZFzniwnSyZfZ7v7na",    # Arts & Theatre segmentId
            # Add other mappings as needed
        }
        
        city_name = city_mapping.get(city.lower(), city)
        tm_category = category_mapping.get(category.lower(), "") if category else ""
        
        base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {
            "apikey": TICKETMASTER_API_KEY,
            "city": city_name,
            "countryCode": "IN",
            "size": 100,  # Number of events to return
        }
        
        if tm_category:
            params["segmentId"] = tm_category
            
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        events = data.get("_embedded", {}).get("events", [])
        
        standardized_events = []
        for event in events:
            # Extract venue information
            venue_info = event.get("_embedded", {}).get("venues", [{}])[0]
            
            # Extract pricing information if available
            price_ranges = event.get("priceRanges", [{}])[0] if event.get("priceRanges") else {}
            
            standardized_event = {
                "event_id": f"tm_{event.get('id')}",
                "source": "ticketmaster",
                "title": event.get("name", ""),
                "description": event.get("description", ""),
                "location": {
                    "city": venue_info.get("city", {}).get("name", city),
                    "venue": venue_info.get("name", ""),
                    "address": venue_info.get("address", {}).get("line1", ""),
                    "lat": venue_info.get("location", {}).get("latitude"),
                    "lon": venue_info.get("location", {}).get("longitude"),
                },
                "datetime": {
                    "start": event.get("dates", {}).get("start", {}).get("dateTime"),
                    "end": None,  # Ticketmaster often doesn't provide end times
                },
                "category": event.get("classifications", [{}])[0].get("segment", {}).get("name", category or "Other"),
                "pricing": {
                    "min": price_ranges.get("min", 0),
                    "max": price_ranges.get("max", 0),
                    "currency": price_ranges.get("currency", "INR")
                },
                "popularity": 0,  # Ticketmaster doesn't provide this directly
                "image_url": next((img.get("url") for img in event.get("images", []) if img.get("ratio") == "16_9"), ""),
                "ticket_url": event.get("url", ""),
                "last_updated": datetime.now().isoformat()
            }
            standardized_events.append(standardized_event)
        
        return standardized_events
        
    except Exception as e:
        logger.error(f"Error fetching Ticketmaster data: {str(e)}")
        return []

# =================== Dataset Ingestion ===================

def download_kaggle_dataset(dataset_name: str, path: str) -> bool:
    """
    Download a dataset from Kaggle
    """
    try:
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        
        # Set Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY
        
        # Use the Kaggle API to download the dataset
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset {dataset_name}: {str(e)}")
        return False

def process_globo_dataset(file_path: str) -> List[Dict]:
    """
    Process the Globo.com News Portal User Interactions dataset
    """
    try:
        logger.info(f"Processing Globo.com dataset from {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Extract useful features - this is a simplified version
        # In reality, you'd do more complex preprocessing
        
        # Group by article/content and count interactions
        content_interactions = df.groupby('content_id').agg({
            'user_id': 'count',  # Count interactions per content
            'session_id': 'nunique',  # Count unique sessions
            'user_id': 'nunique',  # Count unique users
        }).reset_index()
        
        content_interactions.columns = ['content_id', 'total_clicks', 'unique_sessions', 'unique_users']
        
        # Calculate an engagement score
        content_interactions['engagement_score'] = (
            0.5 * content_interactions['total_clicks'] + 
            0.3 * content_interactions['unique_sessions'] + 
            0.2 * content_interactions['unique_users']
        )
        
        # Convert to dictionary format for MongoDB
        result = content_interactions.to_dict('records')
        
        # Add a dataset identifier and timestamp
        for item in result:
            item['dataset_source'] = 'globo'
            item['processed_at'] = datetime.now().isoformat()
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing Globo dataset: {str(e)}")
        return []

def process_event_attendees_dataset(file_path: str) -> List[Dict]:
    """
    Process the Event Recommendation Engine Challenge dataset
    """
    try:
        logger.info(f"Processing Event Attendees dataset from {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Process event attendance patterns
        event_stats = df.groupby('event_id').agg({
            'user_id': 'count',  # Count attendees per event
            'user_id': 'nunique',  # Count unique attendees
            # Add more aggregations as needed
        }).reset_index()
        
        event_stats.columns = ['event_id', 'total_attendees', 'unique_attendees']
        
        # Calculate event popularity score (simplified)
        event_stats['popularity_score'] = event_stats['unique_attendees'] * 1.5
        
        # Process categorical features if available
        if 'event_category' in df.columns:
            # Get category distribution per event
            categories = df.groupby(['event_id', 'event_category']).size().unstack(fill_value=0)
            event_stats = event_stats.merge(categories, on='event_id', how='left')
        
        # Convert to dictionary format for MongoDB
        result = event_stats.to_dict('records')
        
        # Add source and timestamp
        for item in result:
            item['dataset_source'] = 'event_recommendation'
            item['processed_at'] = datetime.now().isoformat()
            
            # Add default values for missing fields that downstream components might need
            item.setdefault('event_title', f"Event {item['event_id']}")
            item.setdefault('event_description', "")
            item.setdefault('event_category', "Other")
            item.setdefault('venue_name', "Unknown Venue")
            item.setdefault('city', "Unknown City")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing Event Attendees dataset: {str(e)}")
        return []

# =================== Data Storage ===================

def save_to_mongodb(data: List[Dict], collection, dedup_key: str = 'event_id') -> int:
    """
    Save data to MongoDB with deduplication
    """
    if not data:
        logger.warning("No data to save to MongoDB")
        return 0
    
    try:
        count = 0
        for item in data:
            # Create a filter based on the deduplication key
            filter_query = {dedup_key: item[dedup_key]} if dedup_key in item else {"_id": item.get("_id", None)}
            
            # If the record doesn't exist, insert it; otherwise, update it
            result = collection.update_one(
                filter_query,
                {"$set": {**item, "updated_at": datetime.now().isoformat()}},
                upsert=True
            )
            
            if result.modified_count > 0 or result.upserted_id is not None:
                count += 1
                
        logger.info(f"Successfully saved {count} items to MongoDB collection {collection.name}")
        return count
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        return 0

# =================== Caching ===================

def get_cache_key(function_name: str, *args, **kwargs) -> str:
    """
    Generate a cache key for the function call
    """
    args_str = json.dumps(args)
    kwargs_str = json.dumps(kwargs, sort_keys=True)
    
    key = f"{function_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key.encode()).hexdigest()

def is_cache_valid(cache_key: str) -> bool:
    """
    Check if a cache entry is still valid based on expiry time
    """
    if cache_key not in CACHE_TIMESTAMP:
        return False
        
    timestamp = CACHE_TIMESTAMP[cache_key]
    current_time = time.time()
    
    return (current_time - timestamp) < CACHE_EXPIRY_SECONDS

def cache_result(cache_key: str, result: Any) -> None:
    """
    Cache a result with the current timestamp
    """
    EVENT_CACHE[cache_key] = result
    CACHE_TIMESTAMP[cache_key] = time.time()

def get_cached_result(cache_key: str) -> Optional[Any]:
    """
    Get a cached result if it's valid
    """
    if cache_key in EVENT_CACHE and is_cache_valid(cache_key):
        return EVENT_CACHE[cache_key]
    return None

# =================== Public Interface ===================

def fetch_events(city: str, category: Optional[str] = None) -> List[Dict]:
    """
    Fetch events from all sources based on city and optional category
    Uses caching to improve performance
    """
    cache_key = get_cache_key("fetch_events", city, category)
    cached_result = get_cached_result(cache_key)
    
    if cached_result is not None:
        logger.info(f"Returning cached events for {city}, category: {category}")
        return cached_result
    
    try:
        logger.info(f"Fetching events for {city}, category: {category}")
        
        # Gather events from various sources
        meraevents_data = fetch_meraevents(city, category)
        bookmyshow_data = scrape_bookmyshow(city, category)
        ticketmaster_data = fetch_ticketmaster(city, category)
        
        # Combine all events
        all_events = meraevents_data + bookmyshow_data + ticketmaster_data
        
        # Save to MongoDB
        if all_events:
            save_to_mongodb(all_events, events_collection)
        
        # Cache the result
        cache_result(cache_key, all_events)
        
        return all_events
        
    except Exception as e:
        logger.error(f"Error in fetch_events: {str(e)}")
        return []

def fetch_events_by_ids(event_ids: List[str]) -> List[Dict]:
    """
    Fetch specific events by their IDs
    """
    cache_key = get_cache_key("fetch_events_by_ids", event_ids)
    cached_result = get_cached_result(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    try:
        events = list(events_collection.find({"event_id": {"$in": event_ids}}))
        
        # Remove MongoDB's _id field for JSON serialization
        for event in events:
            if "_id" in event:
                event["_id"] = str(event["_id"])
                
        cache_result(cache_key, events)
        return events
        
    except Exception as e:
        logger.error(f"Error fetching events by IDs: {str(e)}")
        return []

def get_popular_events(city: str, limit: int = 10) -> List[Dict]:
    """
    Get the most popular events for a given city
    """
    cache_key = get_cache_key("get_popular_events", city, limit)
    cached_result = get_cached_result(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    try:
        # First ensure we have events for this city
        fetch_events(city)
        
        # Query the most popular events
        events = list(events_collection.find(
            {"location.city": {"$regex": city, "$options": "i"}},
            sort=[("popularity", -1)],
            limit=limit
        ))
        
        # Clean up MongoDB _id
        for event in events:
            if "_id" in event:
                event["_id"] = str(event["_id"])
                
        cache_result(cache_key, events)
        return events
        
    except Exception as e:
        logger.error(f"Error getting popular events: {str(e)}")
        return []

def ingest_kaggle_datasets(force_refresh: bool = False) -> bool:
    """
    Download and process Kaggle datasets, then store in MongoDB
    """
    try:
        logger.info("Starting Kaggle dataset ingestion")
        
        # Create directories if they don't exist
        os.makedirs("data/kaggle", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        # Dataset 1: News Portal User Interactions by Globo.com
        globo_dataset = "gspmoreira/news-portal-user-interactions-by-globocom"
        globo_path = "data/kaggle/globo"
        
        # Only download if it doesn't exist or force refresh is True
        if force_refresh or not os.path.exists(globo_path):
            download_kaggle_dataset(globo_dataset, globo_path)
        
        # Process the Globo dataset
        globo_data = process_globo_dataset(os.path.join(globo_path, "clicks.csv"))
        
        # Save to MongoDB
        save_to_mongodb(globo_data, kaggle_collection, "content_id")
        
        # Dataset 2: Event Recommendation Engine Challenge
        event_dataset = "cjgdev/event-recommendation-engine-challenge"
        event_path = "data/kaggle/events"
        
        # Only download if it doesn't exist or force refresh is True
        if force_refresh or not os.path.exists(event_path):
            download_kaggle_dataset(event_dataset, event_path)
        
        # Process the Events dataset
        event_data = process_event_attendees_dataset(os.path.join(event_path, "event_attendees.csv"))
        
        # Save to MongoDB
        save_to_mongodb(event_data, kaggle_collection, "event_id")
        
        logger.info("Kaggle dataset ingestion completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting Kaggle datasets: {str(e)}")
        return False

def refresh_event_data(cities: List[str] = None) -> int:
    """
    Refresh event data for specified cities, or all cities if None
    Returns the count of events refreshed
    """
    try:
        if cities is None:
            # Get a list of all cities we have data for
            all_cities = events_collection.distinct("location.city")
            cities = all_cities
            
        total_events = 0
        
        for city in cities:
            logger.info(f"Refreshing event data for {city}")
            
            # Fetch events for each city (which will update the database)
            events = fetch_events(city)
            total_events += len(events)
            
        logger.info(f"Refreshed data for {len(cities)} cities, total {total_events} events")
        return total_events
        
    except Exception as e:
        logger.error(f"Error refreshing event data: {str(e)}")
        return 0

def search_events(query: str, city: Optional[str] = None, category: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """
    Search for events based on text query, city, and category
    """
    try:
        search_filter = {}
        
        # Add text search
        if query:
            search_filter["$text"] = {"$search": query}
        
        # Add city filter
        if city:
            search_filter["location.city"] = {"$regex": city, "$options": "i"}
            
        # Add category filter
        if category:
            search_filter["category"] = {"$regex": category, "$options": "i"}
            
        # Execute the search
        if "$text" in search_filter:
            # If we're doing a text search, use text score for sorting
            events = list(events_collection.find(
                search_filter,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))
        else:
            # Otherwise, sort by popularity
            events = list(events_collection.find(
                search_filter
            ).sort([("popularity", -1)]).limit(limit))
            
        # Clean up MongoDB _id
        for event in events:
            if "_id" in event:
                event["_id"] = str(event["_id"])
                
        return events
        
    except Exception as e:
        logger.error(f"Error searching events: {str(e)}")
        return []

def get_event_categories() -> List[str]:
    """
    Get a list of all event categories in the database
    """
    try:
        categories = events_collection.distinct("category")
        return categories
    except Exception as e:
        logger.error(f"Error getting event categories: {str(e)}")
        return []

def get_event_cities() -> List[str]:
    """
    Get a list of all cities with events in the database
    """
    try:
        cities = events_collection.distinct("location.city")
        return cities
    except Exception as e:
        logger.error(f"Error getting event cities: {str(e)}")
        return []

# Initialize (create indexes, etc.)
def initialize_module():
    """
    Initialize the data ingestion module
    This would be called when the application starts
    """
    try:
        # Create indexes for better query performance
        events_collection.create_index("event_id", unique=True)
        events_collection.create_index("location.city")
        events_collection.create_index("category")
        events_collection.create_index("datetime.start")
        events_collection.create_index("popularity")
        events_collection.create_index([("title", "text"), ("description", "text")])
        
        kaggle_collection.create_index("event_id", unique=True)
        kaggle_collection.create_index("content_id", unique=True)
        
        logger.info("Data ingestion module initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing data ingestion module: {str(e)}")
        return False