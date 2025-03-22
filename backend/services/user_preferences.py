"""
TufanTicket - User Preferences Service
-------------------------------------
This module manages user preferences and profiles, including:
- Storing and retrieving user preferences
- Logging user interactions with events
- Dynamically updating preferences based on interaction patterns
- Providing personalized recommendations based on user history
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from bson import ObjectId
from functools import lru_cache

# Import models
from ..models.user import UserModel
from ..models.event import EventModel, EventCategory
from ..models.interaction import (
    InteractionModel, 
    InteractionType, 
    SwipeDirection,
    create_interaction, 
    get_user_interactions
)

# MongoDB connection
from pymongo import MongoClient, ASCENDING, DESCENDING
from ..config import MONGODB_URI, MONGODB_DB_NAME, CACHE_EXPIRY_SECONDS

# Setup logging
logger = logging.getLogger(__name__)

# Database connection
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
users_collection = db.users
events_collection = db.events
interactions_collection = db.interactions


class UserPreferenceService:
    """Service for managing user preferences and interactions."""

    @staticmethod
    def get_user_profile(user_id: str) -> Optional[Dict]:
        """
        Retrieve a user's profile including preferences.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary containing user profile or None if not found
        """
        user_obj_id = ObjectId(user_id)
        user = users_collection.find_one({"_id": user_obj_id})
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            return None
            
        # Convert ObjectId to string for serialization
        user["_id"] = str(user["_id"])
        return user

    @staticmethod
    def update_user_preferences(user_id: str, preferences: Dict) -> bool:
        """
        Update a user's preferences directly.
        
        Args:
            user_id: The user's ID
            preferences: Dictionary of preferences to update
            
        Returns:
            Boolean indicating success
        """
        user_obj_id = ObjectId(user_id)
        
        # Validate user exists
        if not users_collection.find_one({"_id": user_obj_id}):
            logger.error(f"Cannot update preferences for non-existent user: {user_id}")
            return False
            
        # Build update document
        update_data = {"$set": {}}
        
        # Handle category preferences
        if "preferred_categories" in preferences:
            # Validate that all categories are valid
            for category in preferences["preferred_categories"]:
                try:
                    # This will raise ValueError if category is invalid
                    EventCategory(category)
                except ValueError:
                    logger.warning(f"Invalid category in preferences update: {category}")
                    return False
                    
            update_data["$set"]["preferred_categories"] = preferences["preferred_categories"]
            
        # Handle location preference
        if "preferred_location" in preferences:
            location = preferences["preferred_location"]
            if "city" in location and "state" in location:
                update_data["$set"]["preferred_location"] = location
            else:
                logger.warning("Location must include city and state")
                return False
                
        # Handle price range
        if "price_range" in preferences:
            price_range = preferences["price_range"]
            if "min" in price_range and "max" in price_range:
                if price_range["min"] > price_range["max"]:
                    logger.warning("Invalid price range: min > max")
                    return False
                update_data["$set"]["price_range"] = price_range
            else:
                logger.warning("Price range must include min and max values")
                return False
                
        # Handle notification preferences
        if "notification_preferences" in preferences:
            update_data["$set"]["notification_preferences"] = preferences["notification_preferences"]
            
        # Only update if there's something to update
        if update_data["$set"]:
            # Add last updated timestamp
            update_data["$set"]["preferences_updated_at"] = datetime.now()
            
            result = users_collection.update_one(
                {"_id": user_obj_id}, 
                update_data
            )
            return result.modified_count > 0
            
        return True  # Nothing to update is still a success

    @staticmethod
    def log_interaction(
        user_id: str, 
        event_id: str, 
        interaction_type: InteractionType, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a user's interaction with an event.
        
        Args:
            user_id: The user's ID
            event_id: The event's ID
            interaction_type: Type of interaction
            metadata: Optional additional data about the interaction
            
        Returns:
            ID of the created interaction record
        """
        if not metadata:
            metadata = {}
            
        # Validate that user and event exist
        user_obj_id = ObjectId(user_id)
        event_obj_id = ObjectId(event_id)
        
        user = users_collection.find_one({"_id": user_obj_id})
        event = events_collection.find_one({"_id": event_obj_id})
        
        if not user:
            logger.error(f"Cannot log interaction for non-existent user: {user_id}")
            raise ValueError(f"User {user_id} not found")
            
        if not event:
            logger.error(f"Cannot log interaction for non-existent event: {event_id}")
            raise ValueError(f"Event {event_id} not found")
        
        # Create interaction record
        interaction_data = {
            "user_id": user_obj_id,
            "event_id": event_obj_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.now(),
            "metadata": metadata
        }
        
        # Special handling for swipes
        if interaction_type in [InteractionType.SWIPE_LEFT, InteractionType.SWIPE_RIGHT]:
            interaction_data["swipe_direction"] = (
                SwipeDirection.RIGHT if interaction_type == InteractionType.SWIPE_RIGHT 
                else SwipeDirection.LEFT
            )
        
        # Record the interaction
        interaction_id = create_interaction(interaction_data)
        
        # Update recommendations and user preferences asynchronously
        UserPreferenceService._update_preferences_from_interaction(
            user_id, event_id, interaction_type, metadata
        )
        
        return interaction_id

    @staticmethod
    def log_swipe(
        user_id: str, 
        event_id: str, 
        direction: SwipeDirection, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a swipe interaction.
        
        Args:
            user_id: The user's ID
            event_id: The event's ID
            direction: Direction of the swipe (left or right)
            metadata: Optional additional data about the swipe
            
        Returns:
            ID of the created interaction record
        """
        interaction_type = (
            InteractionType.SWIPE_RIGHT if direction == SwipeDirection.RIGHT 
            else InteractionType.SWIPE_LEFT
        )
        
        if not metadata:
            metadata = {}
            
        metadata["swipe_direction"] = direction
        
        return UserPreferenceService.log_interaction(
            user_id, event_id, interaction_type, metadata
        )

    @staticmethod
    def _update_preferences_from_interaction(
        user_id: str, 
        event_id: str, 
        interaction_type: InteractionType, 
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update user preferences based on their interactions with events.
        This enables dynamic preference learning over time.
        
        Args:
            user_id: The user's ID
            event_id: The event's ID
            interaction_type: Type of interaction
            metadata: Additional data about the interaction
        """
        # Skip for certain interaction types
        if interaction_type in [InteractionType.VIEW]:
            return
            
        user_obj_id = ObjectId(user_id)
        event_obj_id = ObjectId(event_id)
        
        # Get event details
        event = events_collection.find_one({"_id": event_obj_id})
        if not event:
            logger.error(f"Event not found for preference update: {event_id}")
            return
            
        # Initialize update document
        update_data = {}
        
        # Handle category preference updates for positive interactions
        if interaction_type in [InteractionType.SWIPE_RIGHT, InteractionType.BOOKMARK, InteractionType.REGISTRATION, InteractionType.PURCHASE]:
            if "category" in event:
                # Increment category preference counter
                update_data["$inc"] = {f"category_preferences.{event['category']}": 1}
                
        # For negative interactions, we might want to decrement preference
        if interaction_type in [InteractionType.SWIPE_LEFT]:
            if "category" in event:
                # Decrement category preference, but don't go below 0
                update_data["$inc"] = {f"category_preferences.{event['category']}": -0.5}
                
        # Update engagement metrics
        if interaction_type in [InteractionType.REGISTRATION, InteractionType.PURCHASE]:
            update_data.setdefault("$inc", {})
            update_data["$inc"]["total_registrations"] = 1
            
        # Update last activity timestamp
        update_data["$set"] = {"last_activity": datetime.now()}
        
        # Update price preferences for purchases
        if interaction_type == InteractionType.PURCHASE and "price" in metadata:
            # Set price preference using exponential weighted moving average
            users_collection.update_one(
                {"_id": user_obj_id},
                {
                    "$set": {
                        "price_preference": {
                            "$ifNull": [
                                {
                                    "$add": [
                                        {"$multiply": [{"$ifNull": ["$price_preference", metadata["price"]]}, 0.7]},
                                        {"$multiply": [metadata["price"], 0.3]}
                                    ]
                                },
                                metadata["price"]
                            ]
                        }
                    }
                }
            )
            
        # Apply the updates if there are any
        if update_data:
            users_collection.update_one(
                {"_id": user_obj_id},
                update_data
            )
            
    @staticmethod
    def get_recommendation_data(user_id: str) -> Dict[str, Any]:
        """
        Get user data needed for the recommendation engine.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary containing user preferences and interaction history
        """
        user_obj_id = ObjectId(user_id)
        
        # Get user profile
        user = users_collection.find_one({"_id": user_obj_id})
        if not user:
            logger.error(f"User not found for recommendations: {user_id}")
            raise ValueError(f"User {user_id} not found")
            
        # Get recent interactions (last 90 days)
        ninety_days_ago = datetime.now() - timedelta(days=90)
        
        interactions = list(interactions_collection.find({
            "user_id": user_obj_id,
            "timestamp": {"$gte": ninety_days_ago}
        }).sort("timestamp", DESCENDING))
        
        # Extract liked and disliked event IDs
        liked_events = [
            str(interaction["event_id"]) 
            for interaction in interactions 
            if interaction["interaction_type"] == InteractionType.SWIPE_RIGHT
        ]
        
        disliked_events = [
            str(interaction["event_id"]) 
            for interaction in interactions 
            if interaction["interaction_type"] == InteractionType.SWIPE_LEFT
        ]
        
        # Get category preferences
        category_prefs = user.get("category_preferences", {})
        
        # Convert to list of tuples sorted by preference score
        sorted_categories = sorted(
            category_prefs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "user_id": str(user["_id"]),
            "preferred_categories": [cat for cat, _ in sorted_categories],
            "preferred_location": user.get("preferred_location", {}),
            "price_range": user.get("price_range", {"min": 0, "max": 10000}),
            "liked_events": liked_events,
            "disliked_events": disliked_events,
            "interaction_history": [
                {
                    "event_id": str(interaction["event_id"]),
                    "interaction_type": interaction["interaction_type"],
                    "timestamp": interaction["timestamp"]
                }
                for interaction in interactions[:100]  # Limit to most recent 100
            ]
        }
        
    @staticmethod
    @lru_cache(maxsize=100)
    def get_user_interaction_stats(user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's interactions.
        Results are cached to improve performance.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary of interaction statistics
        """
        user_obj_id = ObjectId(user_id)
        
        # Get count by interaction type
        pipeline = [
            {"$match": {"user_id": user_obj_id}},
            {"$group": {
                "_id": "$interaction_type",
                "count": {"$sum": 1}
            }}
        ]
        
        type_counts = {
            doc["_id"]: doc["count"] 
            for doc in interactions_collection.aggregate(pipeline)
        }
        
        # Get counts by category
        category_pipeline = [
            {
                "$match": {
                    "user_id": user_obj_id,
                    "interaction_type": {"$in": [
                        InteractionType.SWIPE_RIGHT,
                        InteractionType.BOOKMARK,
                        InteractionType.REGISTRATION,
                        InteractionType.PURCHASE
                    ]}
                }
            },
            {
                "$lookup": {
                    "from": "events",
                    "localField": "event_id",
                    "foreignField": "_id",
                    "as": "event"
                }
            },
            {"$unwind": "$event"},
            {"$group": {
                "_id": "$event.category",
                "count": {"$sum": 1}
            }}
        ]
        
        category_counts = {
            doc["_id"]: doc["count"]
            for doc in interactions_collection.aggregate(category_pipeline)
        }
        
        # Calculate conversion rate (views to purchases)
        views = type_counts.get(InteractionType.VIEW, 0)
        purchases = type_counts.get(InteractionType.PURCHASE, 0)
        
        conversion_rate = (purchases / views * 100) if views > 0 else 0
        
        return {
            "interaction_counts": type_counts,
            "category_interests": category_counts,
            "total_interactions": sum(type_counts.values()),
            "conversion_rate": round(conversion_rate, 2),
            "swipe_ratio": {
                "right": type_counts.get(InteractionType.SWIPE_RIGHT, 0),
                "left": type_counts.get(InteractionType.SWIPE_LEFT, 0)
            }
        }
        
    @staticmethod
    def invalidate_cache(user_id: str) -> None:
        """
        Invalidate cached data for a user.
        
        Args:
            user_id: The user's ID
        """
        try:
            UserPreferenceService.get_user_interaction_stats.cache_clear()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")