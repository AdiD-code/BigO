"""
TufanTicket - User Interaction Model
-----------------------------------
This module defines the data models for user interactions with events,
including clicks, swipes, and registrations. It also provides
MongoDB operations for creating, retrieving, and analyzing interactions.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

# Setup MongoDB connection
from pymongo import MongoClient
from ..config import MONGODB_URI, MONGODB_DB_NAME

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
interactions_collection = db.interactions


class InteractionType(str, Enum):
    """Enumeration of possible user interaction types with events."""
    CLICK = "click"               # User clicked on event details
    VIEW = "view"                 # User viewed event details
    SWIPE_RIGHT = "swipe_right"   # User showed interest (similar to like)
    SWIPE_LEFT = "swipe_left"     # User rejected the event
    BOOKMARK = "bookmark"         # User saved the event for later
    SHARE = "share"               # User shared the event
    REGISTRATION = "registration" # User registered for the event
    PURCHASE = "purchase"         # User purchased tickets


class SwipeDirection(str, Enum):
    """Enumeration for swipe directions in the user interface."""
    LEFT = "left"     # Reject/dislike
    RIGHT = "right"   # Accept/like


class InteractionModel(BaseModel):
    """Data model for user interactions with events."""
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    event_id: str
    interaction_type: InteractionType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Optional additional data based on interaction type
    duration_seconds: Optional[int] = None  # For views
    swipe_direction: Optional[SwipeDirection] = None  # For swipes
    source_page: Optional[str] = None  # Where the interaction originated (homepage, search, etc.)
    device_info: Optional[Dict[str, str]] = None  # Device type, OS, browser
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }
        
    @validator('user_id', 'event_id')
    def validate_object_id(cls, v):
        """Ensure ObjectID fields are valid."""
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId format: {v}")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v, values):
        """Add specific validation for metadata based on interaction type."""
        interaction_type = values.get('interaction_type')
        
        if interaction_type == InteractionType.PURCHASE and 'ticket_quantity' not in v:
            raise ValueError("Purchase interactions require 'ticket_quantity' in metadata")
            
        if interaction_type == InteractionType.REGISTRATION and 'registration_status' not in v:
            # Default to 'pending' if not provided
            v['registration_status'] = 'pending'
            
        return v


# MongoDB Operations for Interactions

def create_interaction(interaction_data: dict) -> str:
    """
    Create a new interaction record in the database.
    
    Args:
        interaction_data: Dictionary containing interaction details
        
    Returns:
        str: ID of the created interaction
    """
    # Convert string representation of ObjectId to actual ObjectId if needed
    if 'user_id' in interaction_data and isinstance(interaction_data['user_id'], str):
        interaction_data['user_id'] = ObjectId(interaction_data['user_id'])
        
    if 'event_id' in interaction_data and isinstance(interaction_data['event_id'], str):
        interaction_data['event_id'] = ObjectId(interaction_data['event_id'])
    
    # Ensure timestamp is set if not provided
    if 'timestamp' not in interaction_data:
        interaction_data['timestamp'] = datetime.utcnow()
    
    result = interactions_collection.insert_one(interaction_data)
    return str(result.inserted_id)


def get_user_interactions(user_id: str, limit: int = 100) -> List[Dict]:
    """
    Retrieve interactions for a specific user.
    
    Args:
        user_id: The user's ID
        limit: Maximum number of interactions to return
        
    Returns:
        List of interaction records
    """
    user_obj_id = ObjectId(user_id)
    cursor = interactions_collection.find(
        {"user_id": user_obj_id}
    ).sort("timestamp", DESCENDING).limit(limit)
    
    return [{**doc, "_id": str(doc["_id"])} for doc in cursor]


def get_event_interactions(event_id: str, limit: int = 100) -> List[Dict]:
    """
    Retrieve interactions for a specific event.
    
    Args:
        event_id: The event's ID
        limit: Maximum number of interactions to return
        
    Returns:
        List of interaction records
    """
    event_obj_id = ObjectId(event_id)
    cursor = interactions_collection.find(
        {"event_id": event_obj_id}
    ).sort("timestamp", DESCENDING).limit(limit)
    
    return [{**doc, "_id": str(doc["_id"])} for doc in cursor]


def get_user_event_interaction(user_id: str, event_id: str) -> Optional[Dict]:
    """
    Get a specific user's interactions with a specific event.
    
    Args:
        user_id: The user's ID
        event_id: The event's ID
        
    Returns:
        Dictionary containing the interaction record or None if not found
    """
    user_obj_id = ObjectId(user_id)
    event_obj_id = ObjectId(event_id)
    
    doc = interactions_collection.find_one({
        "user_id": user_obj_id,
        "event_id": event_obj_id
    })
    
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    
    return None


def count_interactions_by_type(event_id: str = None) -> Dict[str, int]:
    """
    Count interactions grouped by interaction type, optionally filtered by event.
    
    Args:
        event_id: Optional event ID to filter interactions
        
    Returns:
        Dictionary mapping interaction types to their counts
    """
    match_stage = {}
    if event_id:
        match_stage["event_id"] = ObjectId(event_id)
    
    pipeline = [
        {"$match": match_stage} if match_stage else {},
        {"$group": {
            "_id": "$interaction_type",
            "count": {"$sum": 1}
        }}
    ]
    
    result = interactions_collection.aggregate(pipeline)
    return {doc["_id"]: doc["count"] for doc in result}


def get_user_liked_events(user_id: str, limit: int = 50) -> List[str]:
    """
    Get events that a user has liked (swiped right on).
    
    Args:
        user_id: The user's ID
        limit: Maximum number of events to return
        
    Returns:
        List of event IDs
    """
    user_obj_id = ObjectId(user_id)
    cursor = interactions_collection.find(
        {
            "user_id": user_obj_id,
            "interaction_type": InteractionType.SWIPE_RIGHT
        },
        {"event_id": 1}
    ).limit(limit)
    
    return [str(doc["event_id"]) for doc in cursor]


def delete_interaction(interaction_id: str) -> bool:
    """
    Delete an interaction from the database.
    
    Args:
        interaction_id: The ID of the interaction to delete
        
    Returns:
        bool: True if successfully deleted, False otherwise
    """
    result = interactions_collection.delete_one({"_id": ObjectId(interaction_id)})
    return result.deleted_count > 0


# Create indexes for faster queries
def setup_indexes():
    """Set up necessary indexes for the interactions collection."""
    interactions_collection.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    interactions_collection.create_index([("event_id", ASCENDING), ("timestamp", DESCENDING)])
    interactions_collection.create_index([("user_id", ASCENDING), ("event_id", ASCENDING)])
    interactions_collection.create_index([("interaction_type", ASCENDING)])
    interactions_collection.create_index([
        ("user_id", ASCENDING), 
        ("interaction_type", ASCENDING), 
        ("timestamp", DESCENDING)
    ])

# Create indexes when module is imported
setup_indexes()