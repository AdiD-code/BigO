# backend/models/event.py

from datetime import datetime
from typing import Dict, List, Optional, Union
from bson import ObjectId
from pydantic import BaseModel, Field, validator, root_validator
import json
from enum import Enum
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from config import MONGODB_URI, MONGODB_DB_NAME, DEFAULT_EVENT_IMAGE

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
events_collection = db.events

class PyObjectId(ObjectId):
    """Custom ObjectId class for Pydantic models compatibility."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class EventSource(str, Enum):
    """Enumeration of event data sources."""
    MERAEVENTS = "meraevents"
    BOOKMYSHOW = "bookmyshow"
    TICKETMASTER = "ticketmaster"
    MANUAL = "manual"
    KAGGLE = "kaggle"
    OTHER = "other"

class EventCategory(str, Enum):
    """Enumeration of event categories."""
    MUSIC = "Music"
    COMEDY = "Comedy"
    THEATRE = "Theatre"
    SPORTS = "Sports"
    CONFERENCE = "Conference"
    WORKSHOP = "Workshop"
    EXHIBITION = "Exhibition"
    FESTIVAL = "Festival"
    NETWORKING = "Networking"
    FOOD = "Food & Drink"
    HEALTH = "Health & Wellness"
    TECHNOLOGY = "Technology"
    BUSINESS = "Business"
    EDUCATION = "Education"
    FAMILY = "Family"
    CHARITY = "Charity & Causes"
    OTHER = "Other"

class LocationModel(BaseModel):
    """Model for event location details."""
    city: str
    venue: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    state: Optional[str] = None
    country: Optional[str] = "India"
    postal_code: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "city": "Mumbai",
                "venue": "Nesco Exhibition Centre",
                "address": "Western Express Highway, Goregaon East",
                "lat": 19.1523,
                "lon": 72.8724,
                "state": "Maharashtra",
                "country": "India",
                "postal_code": "400063"
            }
        }

class DateTimeModel(BaseModel):
    """Model for event date and time details."""
    start: datetime
    end: Optional[datetime] = None
    doors_open: Optional[datetime] = None
    timezone: Optional[str] = "Asia/Kolkata"

    @validator('end')
    def end_after_start(cls, v, values):
        if v and 'start' in values and v < values['start']:
            raise ValueError("End time must be after start time")
        return v

    class Config:
        schema_extra = {
            "example": {
                "start": "2023-08-15T18:00:00",
                "end": "2023-08-15T22:00:00",
                "doors_open": "2023-08-15T17:00:00",
                "timezone": "Asia/Kolkata"
            }
        }

class PricingModel(BaseModel):
    """Model for event pricing details."""
    min: float = 0
    max: Optional[float] = None
    currency: str = "INR"
    tiers: Optional[List[Dict[str, Union[str, float]]]] = None

    @validator('max')
    def max_greater_than_min(cls, v, values):
        if v is not None and 'min' in values and v < values['min']:
            raise ValueError("Max price must be greater than or equal to min price")
        return v

    class Config:
        schema_extra = {
            "example": {
                "min": 500,
                "max": 2000,
                "currency": "INR",
                "tiers": [
                    {"name": "Early Bird", "price": 500},
                    {"name": "Regular", "price": 1000},
                    {"name": "VIP", "price": 2000}
                ]
            }
        }

class OrganizerModel(BaseModel):
    """Model for event organizer details."""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "name": "BookMyShow",
                "email": "support@bookmyshow.com",
                "phone": "+91-22-12345678",
                "website": "https://bookmyshow.com",
                "logo_url": "https://example.com/logo.png",
                "description": "India's largest entertainment ticketing platform"
            }
        }

class EventModel(BaseModel):
    """Base model for events with all details."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    event_id: str
    source: EventSource
    title: str
    description: Optional[str] = None
    location: LocationModel
    datetime: DateTimeModel
    category: Union[EventCategory, str] = EventCategory.OTHER
    sub_category: Optional[str] = None
    pricing: Optional[PricingModel] = None
    popularity: Optional[float] = 0
    image_url: Optional[str] = DEFAULT_EVENT_IMAGE
    ticket_url: Optional[str] = None
    organizer: Optional[OrganizerModel] = None
    tags: Optional[List[str]] = []
    is_featured: Optional[bool] = False
    is_cancelled: Optional[bool] = False
    is_sold_out: Optional[bool] = False
    capacity: Optional[int] = None
    attendees_count: Optional[int] = None
    min_age: Optional[int] = None
    last_updated: Optional[datetime] = Field(default_factory=datetime.now)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    additional_info: Optional[Dict] = {}

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
        schema_extra = {
            "example": {
                "event_id": "me_123456",
                "source": "meraevents",
                "title": "Comic Con India 2023",
                "description": "India's greatest pop culture experience",
                "location": {
                    "city": "Mumbai",
                    "venue": "Jio World Convention Centre",
                    "address": "G Block BKC, Bandra Kurla Complex",
                    "lat": 19.0654,
                    "lon": 72.8660,
                    "state": "Maharashtra",
                    "country": "India"
                },
                "datetime": {
                    "start": "2023-09-20T10:00:00",
                    "end": "2023-09-20T20:00:00"
                },
                "category": "Exhibition",
                "pricing": {
                    "min": 799,
                    "max": 2999,
                    "currency": "INR"
                },
                "popularity": 95.7,
                "image_url": "https://example.com/comiccon.jpg",
                "ticket_url": "https://meraevents.com/event/comic-con-2023",
                "tags": ["comics", "cosplay", "entertainment", "pop culture"]
            }
        }

    @validator('category', pre=True)
    def validate_category(cls, v):
        """Validate and convert category string to enum if possible."""
        if isinstance(v, str):
            try:
                return EventCategory(v)
            except ValueError:
                # If not a valid enum value, return as custom string category
                return v
        return v

    def to_json(self) -> str:
        """Convert the model to JSON string."""
        return json.dumps(self.dict(by_alias=True), default=str)

    def to_dict(self) -> dict:
        """Convert the model to a Python dictionary."""
        return {k: v for k, v in self.dict(by_alias=True).items() if v is not None}

class EventCreate(BaseModel):
    """Model for creating a new event (subset of fields)."""
    title: str
    description: Optional[str] = None
    location: LocationModel
    datetime: DateTimeModel
    category: Union[EventCategory, str] = EventCategory.OTHER
    sub_category: Optional[str] = None
    pricing: Optional[PricingModel] = None
    image_url: Optional[str] = DEFAULT_EVENT_IMAGE
    ticket_url: Optional[str] = None
    organizer: Optional[OrganizerModel] = None
    tags: Optional[List[str]] = []
    min_age: Optional[int] = None
    capacity: Optional[int] = None
    additional_info: Optional[Dict] = {}

class EventUpdate(BaseModel):
    """Model for updating an existing event (all fields optional)."""
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[LocationModel] = None
    datetime: Optional[DateTimeModel] = None
    category: Optional[Union[EventCategory, str]] = None
    sub_category: Optional[str] = None
    pricing: Optional[PricingModel] = None
    popularity: Optional[float] = None
    image_url: Optional[str] = None
    ticket_url: Optional[str] = None
    organizer: Optional[OrganizerModel] = None
    tags: Optional[List[str]] = None
    is_featured: Optional[bool] = None
    is_cancelled: Optional[bool] = None
    is_sold_out: Optional[bool] = None
    capacity: Optional[int] = None
    attendees_count: Optional[int] = None
    min_age: Optional[int] = None
    additional_info: Optional[Dict] = None

class EventResponse(BaseModel):
    """Model for API responses with event data."""
    id: str = Field(..., alias="_id")
    event_id: str
    source: str
    title: str
    description: Optional[str] = None
    location: LocationModel
    datetime: DateTimeModel
    category: str
    sub_category: Optional[str] = None
    pricing: Optional[PricingModel] = None
    popularity: Optional[float] = 0
    image_url: Optional[str] = DEFAULT_EVENT_IMAGE
    ticket_url: Optional[str] = None
    organizer: Optional[OrganizerModel] = None
    tags: Optional[List[str]] = []
    is_featured: Optional[bool] = False
    is_cancelled: Optional[bool] = False
    is_sold_out: Optional[bool] = False
    last_updated: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
        schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "event_id": "me_123456",
                "source": "meraevents",
                "title": "Comic Con India 2023",
                "description": "India's greatest pop culture experience",
                "location": {
                    "city": "Mumbai",
                    "venue": "Jio World Convention Centre"
                },
                "datetime": {
                    "start": "2023-09-20T10:00:00",
                    "end": "2023-09-20T20:00:00"
                },
                "category": "Exhibition",
                "pricing": {
                    "min": 799,
                    "max": 2999,
                    "currency": "INR"
                },
                "image_url": "https://example.com/comiccon.jpg",
                "ticket_url": "https://meraevents.com/event/comic-con-2023"
            }
        }

class EventSearchParams(BaseModel):
    """Model for event search query parameters."""
    query: Optional[str] = None
    city: Optional[str] = None
    category: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    tags: Optional[List[str]] = None
    sort_by: Optional[str] = "popularity"  # popularity, date, price
    sort_order: Optional[str] = "desc"  # asc, desc
    limit: Optional[int] = 20
    offset: Optional[int] = 0

    class Config:
        schema_extra = {
            "example": {
                "query": "music festival",
                "city": "Mumbai",
                "category": "Music",
                "start_date": "2023-08-01T00:00:00",
                "end_date": "2023-08-31T23:59:59",
                "min_price": 500,
                "max_price": 2000,
                "tags": ["rock", "live music"],
                "sort_by": "date",
                "sort_order": "asc",
                "limit": 10,
                "offset": 0
            }
        }

# MongoDB Indexes Setup
def setup_indexes():
    """Set up MongoDB indexes for the events collection."""
    try:
        # Create text indexes for searching
        events_collection.create_index([("title", TEXT), ("description", TEXT), ("tags", TEXT)])
        
        # Create indexes for common queries
        events_collection.create_index("event_id", unique=True)
        events_collection.create_index("source")
        events_collection.create_index("category")
        events_collection.create_index("location.city")
        events_collection.create_index([("datetime.start", ASCENDING)])
        events_collection.create_index([("popularity", DESCENDING)])
        events_collection.create_index([("pricing.min", ASCENDING)])
        events_collection.create_index("is_featured")
        events_collection.create_index("tags")
        
        # Compound indexes for common combined queries
        events_collection.create_index([
            ("location.city", ASCENDING),
            ("category", ASCENDING),
            ("datetime.start", ASCENDING)
        ])
        
        events_collection.create_index([
            ("is_cancelled", ASCENDING),
            ("is_sold_out", ASCENDING),
            ("datetime.start", ASCENDING)
        ])
        
        return True
    except Exception as e:
        print(f"Error setting up MongoDB indexes: {str(e)}")
        return False

# Helper functions for event operations
def get_event_by_id(event_id: str) -> Optional[dict]:
    """Retrieve a specific event by its ID."""
    return events_collection.find_one({"event_id": event_id})

def get_event_by_mongodb_id(object_id: Union[str, ObjectId]) -> Optional[dict]:
    """Retrieve a specific event by MongoDB ObjectID."""
    if isinstance(object_id, str):
        object_id = ObjectId(object_id)
    return events_collection.find_one({"_id": object_id})

def create_event(event_data: dict) -> str:
    """Create a new event and return its MongoDB ID."""
    # Ensure event_id is unique
    if "event_id" not in event_data:
        # Generate a unique event ID if not provided
        event_data["event_id"] = f"custom_{ObjectId()}"
    
    # Set timestamps
    now = datetime.now()
    event_data["created_at"] = now
    event_data["last_updated"] = now
    
    # Set default source if not provided
    if "source" not in event_data:
        event_data["source"] = EventSource.MANUAL.value
    
    # Insert into database
    result = events_collection.insert_one(event_data)
    return str(result.inserted_id)

def update_event(event_id: str, update_data: dict) -> bool:
    """Update an existing event by its ID."""
    # Update the last_updated timestamp
    update_data["last_updated"] = datetime.now()
    
    # Perform the update
    result = events_collection.update_one(
        {"event_id": event_id},
        {"$set": update_data}
    )
    
    return result.modified_count > 0

def delete_event(event_id: str) -> bool:
    """Delete an event by its ID."""
    result = events_collection.delete_one({"event_id": event_id})
    return result.deleted_count > 0

def search_events(params: EventSearchParams) -> List[dict]:
    """Search for events based on various criteria."""
    query = {}
    
    # Text search
    if params.query:
        query["$text"] = {"$search": params.query}
    
    # City filter
    if params.city:
        query["location.city"] = {"$regex": params.city, "$options": "i"}
    
    # Category filter
    if params.category:
        query["category"] = {"$regex": params.category, "$options": "i"}
    
    # Date range filter
    date_query = {}
    if params.start_date:
        date_query["$gte"] = params.start_date
    if params.end_date:
        date_query["$lte"] = params.end_date
    if date_query:
        query["datetime.start"] = date_query
    
    # Price range filter
    price_query = {}
    if params.min_price is not None:
        price_query["$gte"] = params.min_price
    if params.max_price is not None:
        price_query["$lte"] = params.max_price
    if price_query:
        query["pricing.min"] = price_query
    
    # Tags filter
    if params.tags and len(params.tags) > 0:
        query["tags"] = {"$in": params.tags}
    
    # Only show active events by default
    query["is_cancelled"] = {"$ne": True}
    
    # Determine sort field and order
    sort_direction = ASCENDING if params.sort_order == "asc" else DESCENDING
    sort_field = "popularity"
    
    if params.sort_by == "date":
        sort_field = "datetime.start"
    elif params.sort_by == "price":
        sort_field = "pricing.min"
    
    # Execute query
    cursor = events_collection.find(query).sort(sort_field, sort_direction)
    
    # Apply pagination
    if params.offset:
        cursor = cursor.skip(params.offset)
    if params.limit:
        cursor = cursor.limit(params.limit)
    
    # Convert ObjectId to string for each document
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    return results

def get_trending_events(city: Optional[str] = None, limit: int = 10) -> List[dict]:
    """Get trending events based on popularity."""
    query = {"is_cancelled": {"$ne": True}}
    
    if city:
        query["location.city"] = {"$regex": city, "$options": "i"}
    
    # Get events happening soon
    start_date = datetime.now()
    end_date = datetime.now() + timedelta(days=30)  # Next 30 days
    query["datetime.start"] = {"$gte": start_date, "$lte": end_date}
    
    # Sort by popularity and return top events
    cursor = events_collection.find(query).sort("popularity", DESCENDING).limit(limit)
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    return results

def get_upcoming_events(city: Optional[str] = None, category: Optional[str] = None, limit: int = 10) -> List[dict]:
    """Get upcoming events based on date."""
    query = {
        "is_cancelled": {"$ne": True},
        "datetime.start": {"$gte": datetime.now()}
    }
    
    if city:
        query["location.city"] = {"$regex": city, "$options": "i"}
    
    if category:
        query["category"] = {"$regex": category, "$options": "i"}
    
    # Sort by date and return the soonest events
    cursor = events_collection.find(query).sort("datetime.start", ASCENDING).limit(limit)
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    return results

def get_event_statistics() -> dict:
    """Get statistics about events in the database."""
    stats = {
        "total_events": events_collection.count_documents({}),
        "upcoming_events": events_collection.count_documents({
            "datetime.start": {"$gte": datetime.now()}
        }),
        "cancelled_events": events_collection.count_documents({
            "is_cancelled": True
        }),
        "sold_out_events": events_collection.count_documents({
            "is_sold_out": True
        }),
        "featured_events": events_collection.count_documents({
            "is_featured": True
        }),
        "events_by_category": {},
        "events_by_city": {},
        "events_by_source": {}
    }
    
    # Get counts by category
    categories = events_collection.distinct("category")
    for category in categories:
        count = events_collection.count_documents({"category": category})
        stats["events_by_category"][category] = count
    
    # Get counts by city
    cities = events_collection.distinct("location.city")
    for city in cities:
        count = events_collection.count_documents({"location.city": city})
        stats["events_by_city"][city] = count
    
    # Get counts by source
    sources = events_collection.distinct("source")
    for source in sources:
        count = events_collection.count_documents({"source": source})
        stats["events_by_source"][source] = count
    
    return stats

# Initialize MongoDB indexes when module is imported
setup_indexes()