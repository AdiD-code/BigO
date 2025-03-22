"""
recommendation.py - AI-powered event recommendation engine for TufanTicket

This module implements a hybrid recommendation system that combines:
1. Collaborative filtering: Recommends events based on similar users' preferences
2. Content-based filtering: Recommends events based on user's past interactions
3. K-Means clustering: Segments users with similar preferences for better recommendations

The engine processes user preferences, past bookings, swipe history, and interaction
patterns to generate personalized event recommendations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache

import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from backend.config import RECOMMENDATION_ALGORITHM as RECOMMENDATION_CACHE_TTL
from backend.config import RECOMMENDATION_ALGORITHM as RECOMMENDATION_WEIGHTS
from backend.config import MAX_RECOMMENDATIONS
from backend.config import SIMILAR_USERS_LIMIT
from backend.config import ENABLE_USER_CLUSTERING 
from backend.config import MIN_INTERACTION_THRESHOLD

# Local imports
from backend.config import (
    MONGODB_URI, 
    MONGODB_DB_NAME,
    RECOMMENDATION_CACHE_TTL,
    RECOMMENDATION_WEIGHTS,
    MAX_RECOMMENDATIONS,
    SIMILAR_USERS_LIMIT,
    ENABLE_USER_CLUSTERING,
    MIN_INTERACTION_THRESHOLD,
)
from backend.models.event import EventModel
from backend.models.user import UserModel
from backend.models.interaction import InteractionType, InteractionModel


# Configure logging
logger = logging.getLogger(__name__)


class RecommendationScore(BaseModel):
    """Model representing a recommendation score for an event."""
    event_id: str
    score: float
    recommendation_type: str = "hybrid"  # Options: "collaborative", "content_based", "hybrid"
    score_components: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserFeatureVector(BaseModel):
    """Model representing a user's feature vector for recommendations."""
    user_id: str
    category_preferences: Dict[str, float] = Field(default_factory=dict)
    price_range: Tuple[float, float] = (0, 1000)
    location_city: str = ""
    location_coords: Optional[Tuple[float, float]] = None
    cluster_id: Optional[int] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class RecommendationEngine:
    """Main recommendation engine that combines multiple recommendation approaches."""
    
    def __init__(self):
        """Initialize the recommendation engine with database connection."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        self.events_collection = self.db.events
        self.users_collection = self.db.users
        self.interactions_collection = self.db.interactions
        self.recommendations_collection = self.db.recommendations
        self.user_features_collection = self.db.user_features
        
        # Load configuration weights
        self.weights = RECOMMENDATION_WEIGHTS if RECOMMENDATION_WEIGHTS else {
            "category_match": 0.3,
            "location_proximity": 0.2,
            "price_match": 0.15,
            "popularity": 0.1,
            "recency": 0.05,
            "collaborative": 0.2
        }
        
        # Configure KMeans clusters
        self.kmeans = None
        self.n_clusters = 5  # Default number of clusters
        self.scaler = StandardScaler()
        
        # Initialize indexes
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Set up MongoDB indexes for performance optimization."""
        # Index for recommendations
        self.recommendations_collection.create_index([
            ("user_id", 1),
            ("timestamp", -1)
        ])
        
        # Index for user features
        self.user_features_collection.create_index([
            ("user_id", 1),
            ("last_updated", -1)
        ])
        self.user_features_collection.create_index([
            ("cluster_id", 1)
        ])
    
    def generate_recommendations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Generate personalized event recommendations for a given user.
        
        Args:
            user_id: The ID of the user to generate recommendations for
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended events with scores
        """
        try:
            # Check if we have recent cached recommendations
            cached_recommendations = self._get_cached_recommendations(user_id, limit)
            if cached_recommendations:
                logger.info(f"Retrieved cached recommendations for user {user_id}")
                return cached_recommendations
            
            # Get user data and check if we have enough interactions
            user_data = self._get_user_data(user_id)
            if not user_data:
                logger.warning(f"User {user_id} not found, returning trending events")
                return self._get_fallback_recommendations(limit)
            
            # Extract user interactions
            user_interactions = self._get_user_interactions(user_id)
            if len(user_interactions) < MIN_INTERACTION_THRESHOLD:
                logger.info(f"User {user_id} has insufficient interactions, using partial personalization")
                
            # Get or create user feature vector
            user_features = self._get_or_create_user_features(user_id, user_data, user_interactions)
            
            # Apply clustering if enabled and we have enough users
            if ENABLE_USER_CLUSTERING:
                self._update_user_clustering(user_id, user_features)
            
            # Get candidate events (events not yet interacted with)
            candidate_events = self._get_candidate_events(user_id, user_interactions)
            if not candidate_events:
                logger.warning(f"No candidate events found for user {user_id}")
                return self._get_fallback_recommendations(limit)
            
            # Generate scores using hybrid approach
            event_scores = self._score_events(
                user_id=user_id,
                user_data=user_data,
                user_features=user_features,
                candidate_events=candidate_events,
                user_interactions=user_interactions
            )
            
            # Sort and limit recommendations
            top_recommendations = sorted(
                event_scores, 
                key=lambda x: x["score"], 
                reverse=True
            )[:limit]
            
            # Store recommendations
            self._store_recommendations(user_id, top_recommendations)
            
            # Return recommendation results
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return self._get_fallback_recommendations(limit)
    
    def _get_cached_recommendations(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get cached recommendations if they exist and are recent."""
        cache_threshold = datetime.now() - timedelta(seconds=RECOMMENDATION_CACHE_TTL)
        
        cached_recs = list(self.recommendations_collection.find(
            {
                "user_id": user_id,
                "timestamp": {"$gt": cache_threshold}
            }
        ).sort("score", -1).limit(limit))
        
        if cached_recs:
            # Convert ObjectId to string for each recommendation
            for rec in cached_recs:
                if "_id" in rec:
                    rec["_id"] = str(rec["_id"])
            
            # Get full event details for recommendations
            event_ids = [rec["event_id"] for rec in cached_recs]
            events = self._get_events_by_ids(event_ids)
            
            # Map events to recommendations
            event_map = {str(event["_id"]): event for event in events}
            for rec in cached_recs:
                if rec["event_id"] in event_map:
                    rec["event"] = event_map[rec["event_id"]]
            
            return cached_recs
        
        return []
    
    def _get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get user data from database."""
        return self.users_collection.find_one({"_id": ObjectId(user_id)})
    
    def _get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interactions from database."""
        # Get interactions from the last 90 days
        cutoff_date = datetime.now() - timedelta(days=90)
        
        return list(self.interactions_collection.find({
            "user_id": user_id,
            "timestamp": {"$gt": cutoff_date}
        }).sort("timestamp", -1))
    
    def _get_or_create_user_features(
        self, 
        user_id: str, 
        user_data: Dict[str, Any], 
        user_interactions: List[Dict[str, Any]]
    ) -> UserFeatureVector:
        """
        Get existing user feature vector or create a new one based on user data and interactions.
        
        This builds a feature vector that represents user preferences for recommendation calculations.
        """
        # Check if we have a recent feature vector
        existing_features = self.user_features_collection.find_one({
            "user_id": user_id
        })
        
        # If we have recent features, use those
        if existing_features and existing_features.get("last_updated", datetime.min) > datetime.now() - timedelta(days=1):
            return UserFeatureVector(**existing_features)
        
        # Otherwise, create new features
        category_counts = {}
        for interaction in user_interactions:
            # Get event details for this interaction
            event_id = interaction.get("event_id")
            if not event_id:
                continue
                
            event = self.events_collection.find_one({"_id": ObjectId(event_id)})
            if not event:
                continue
                
            # Increment category counts based on interaction type
            interaction_type = interaction.get("interaction_type", "")
            category = event.get("category", "")
            
            # Skip if no category
            if not category:
                continue
                
            # Weigh different interaction types differently
            weight = 1.0
            if interaction_type == InteractionType.swipe_right.value:
                weight = 3.0
            elif interaction_type == InteractionType.bookmark.value:
                weight = 5.0
            elif interaction_type == InteractionType.purchase.value:
                weight = 10.0
            
            # Update category counts
            if category in category_counts:
                category_counts[category] += weight
            else:
                category_counts[category] = weight
        
        # Normalize category preferences
        total_weight = sum(category_counts.values()) if category_counts else 1
        category_preferences = {cat: count/total_weight for cat, count in category_counts.items()}
        
        # Get user location
        location_city = user_data.get("preferences", {}).get("location", {}).get("city", "")
        location_coords = None
        if "location" in user_data.get("preferences", {}):
            coords = user_data.get("preferences", {}).get("location", {}).get("coordinates")
            if coords and len(coords) == 2:
                location_coords = (coords[0], coords[1])
        
        # Get price range
        price_preferences = user_data.get("preferences", {}).get("price_range", {})
        min_price = price_preferences.get("min", 0)
        max_price = price_preferences.get("max", 1000)
        
        # Create feature vector
        features = UserFeatureVector(
            user_id=user_id,
            category_preferences=category_preferences,
            price_range=(min_price, max_price),
            location_city=location_city,
            location_coords=location_coords,
            last_updated=datetime.now()
        )
        
        # Store in database
        self.user_features_collection.update_one(
            {"user_id": user_id},
            {"$set": features.dict()},
            upsert=True
        )
        
        return features
    
    def _update_user_clustering(self, user_id: str, user_features: UserFeatureVector):
        """
        Update user clustering model if needed, and assign cluster to the user.
        
        Uses K-Means clustering to group similar users based on their preferences.
        """
        # Skip if we already have a recent clustering model
        feature_count = self.user_features_collection.count_documents({})
        
        # Only perform clustering if we have enough users
        if feature_count < 20:  # Arbitrary threshold, adjust as needed
            return
        
        # Get all user features
        all_features = list(self.user_features_collection.find({}))
        
        # Extract features for clustering
        feature_matrix = []
        user_ids = []
        
        for feature in all_features:
            # Create a feature vector for each user
            feature_vector = []
            
            # Add category preferences
            categories = list(feature.get("category_preferences", {}).keys())
            all_categories = set(categories)
            for category in all_categories:
                feature_vector.append(feature.get("category_preferences", {}).get(category, 0))
            
            # Add price range
            price_range = feature.get("price_range", (0, 1000))
            feature_vector.append(price_range[0])  # Min price
            feature_vector.append(price_range[1])  # Max price
            
            # Skip if we don't have enough features
            if len(feature_vector) < 3:
                continue
                
            feature_matrix.append(feature_vector)
            user_ids.append(feature["user_id"])
        
        # Convert to numpy array
        feature_matrix = np.array(feature_matrix)
        
        # Skip if we don't have enough users with complete features
        if len(feature_matrix) < 10:
            return
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Determine optimal number of clusters (simplified)
        n_clusters = min(5, len(scaled_features) // 5)  # 1 cluster per 5 users, max 5
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Store the model for reuse
        self.kmeans = kmeans
        
        # Update cluster assignments for all users
        for i, user_id in enumerate(user_ids):
            self.user_features_collection.update_one(
                {"user_id": user_id},
                {"$set": {"cluster_id": int(cluster_labels[i])}}
            )
    
    def _get_candidate_events(
        self, 
        user_id: str, 
        user_interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get candidate events that the user hasn't interacted with.
        
        Filters out events that the user has already interacted with to avoid
        recommending the same event multiple times.
        """
        # Extract event IDs the user has already interacted with
        interacted_event_ids = set()
        for interaction in user_interactions:
            event_id = interaction.get("event_id")
            if event_id:
                interacted_event_ids.add(ObjectId(event_id))
        
        # Find events not yet interacted with
        now = datetime.now()
        query = {
            "_id": {"$nin": list(interacted_event_ids)},
            "datetime.start": {"$gt": now}  # Only future events
        }
        
        return list(self.events_collection.find(query).limit(100))  # Limit to 100 candidates for efficiency
    
    def _score_events(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        user_features: UserFeatureVector,
        candidate_events: List[Dict[str, Any]],
        user_interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score candidate events using hybrid filtering approach.
        
        Combines content-based filtering (user preferences) and collaborative
        filtering (similar users' preferences) to score events.
        """
        event_scores = []
        
        # Get collaborative filtering recommendations
        collaborative_scores = self._collaborative_filtering(user_id, user_features)
        
        for event in candidate_events:
            event_id = str(event["_id"])
            score_components = {}
            
            # 1. Content-based scoring components
            
            # Category match
            event_category = event.get("category", "")
            category_score = user_features.category_preferences.get(event_category, 0)
            score_components["category_match"] = category_score * self.weights["category_match"]
            
            # Location proximity
            location_score = 0
            event_city = event.get("location", {}).get("city", "")
            
            if event_city and user_features.location_city and event_city.lower() == user_features.location_city.lower():
                location_score = 1.0
            score_components["location_proximity"] = location_score * self.weights["location_proximity"]
            
            # Price match
            price_score = 0
            event_price = event.get("ticket_price", {}).get("base_price", 0)
            
            if event_price and user_features.price_range:
                min_price, max_price = user_features.price_range
                if min_price <= event_price <= max_price:
                    price_score = 1.0
                elif event_price < min_price:
                    # Below minimum but still affordable
                    price_score = 0.7
                else:
                    # Above maximum
                    price_score = max(0, 1 - (event_price - max_price) / max_price)
            
            score_components["price_match"] = price_score * self.weights["price_match"]
            
            # Popularity factor
            popularity_score = min(1.0, event.get("views_count", 0) / 1000)
            score_components["popularity"] = popularity_score * self.weights["popularity"]
            
            # Recency factor
            recency_score = 0
            if "created_at" in event:
                days_old = (datetime.now() - event["created_at"]).days
                recency_score = max(0, 1 - (days_old / 30))  # Newer is better
            score_components["recency"] = recency_score * self.weights["recency"]
            
            # 2. Collaborative filtering component
            collab_score = collaborative_scores.get(event_id, 0)
            score_components["collaborative"] = collab_score * self.weights["collaborative"]
            
            # Calculate final score
            final_score = sum(score_components.values())
            
            # Create recommendation entry
            recommendation = {
                "event_id": event_id,
                "score": final_score,
                "score_components": score_components,
                "event": event
            }
            
            event_scores.append(recommendation)
        
        return event_scores
    
    @lru_cache(maxsize=100)
    def _collaborative_filtering(
        self, 
        user_id: str, 
        user_features: UserFeatureVector
    ) -> Dict[str, float]:
        """
        Implement collaborative filtering to find events liked by similar users.
        
        Uses clustering and user similarity to identify events that similar users
        have interacted with positively.
        """
        event_scores = {}
        
        try:
            # Find similar users from the same cluster if possible
            similar_users = []
            
            if user_features.cluster_id is not None:
                # Find users in the same cluster
                cluster_users = list(self.user_features_collection.find(
                    {"cluster_id": user_features.cluster_id, "user_id": {"$ne": user_id}}
                ).limit(SIMILAR_USERS_LIMIT))
                
                similar_users.extend(cluster_users)
            
            # If we don't have enough similar users from clustering, find more
            if len(similar_users) < SIMILAR_USERS_LIMIT:
                # Get additional similar users by category preferences
                category_prefs_vector = []
                for cat, score in user_features.category_preferences.items():
                    category_prefs_vector.append(score)
                
                if category_prefs_vector:
                    # Find users with feature vectors
                    other_users = list(self.user_features_collection.find(
                        {"user_id": {"$ne": user_id}}
                    ))
                    
                    # Compute similarities
                    user_similarities = []
                    for other_user in other_users:
                        if other_user["user_id"] in [u["user_id"] for u in similar_users]:
                            continue  # Skip users already in similar_users
                        
                        other_prefs = []
                        for cat, score in other_user.get("category_preferences", {}).items():
                            other_prefs.append(score)
                        
                        if other_prefs:
                            # Calculate Jaccard similarity of category preferences
                            intersection = len(set(user_features.category_preferences.keys()) & 
                                              set(other_user.get("category_preferences", {}).keys()))
                            union = len(set(user_features.category_preferences.keys()) | 
                                       set(other_user.get("category_preferences", {}).keys()))
                            
                            similarity = intersection / union if union > 0 else 0
                            
                            user_similarities.append((other_user, similarity))
                    
                    # Sort by similarity
                    user_similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add top similar users
                    additional_needed = SIMILAR_USERS_LIMIT - len(similar_users)
                    for user, _ in user_similarities[:additional_needed]:
                        similar_users.append(user)
            
            # Get positive interactions from similar users
            for similar_user in similar_users:
                similar_user_id = similar_user["user_id"]
                
                # Get swipe_right, bookmark, and purchase interactions
                positive_interactions = list(self.interactions_collection.find({
                    "user_id": similar_user_id,
                    "interaction_type": {"$in": [
                        InteractionType.swipe_right.value,
                        InteractionType.bookmark.value,
                        InteractionType.purchase.value
                    ]}
                }))
                
                # Score events based on interaction type and recency
                for interaction in positive_interactions:
                    event_id = interaction.get("event_id")
                    if not event_id:
                        continue
                    
                    # Calculate score based on interaction type
                    interaction_weight = 1.0
                    if interaction["interaction_type"] == InteractionType.bookmark.value:
                        interaction_weight = 3.0
                    elif interaction["interaction_type"] == InteractionType.purchase.value:
                        interaction_weight = 5.0
                    
                    # Include recency factor
                    days_old = (datetime.now() - interaction.get("timestamp", datetime.now())).days
                    recency_factor = max(0.5, 1 - (days_old / 30))  # Older interactions less important
                    
                    # Get similarity factor if available
                    similarity_factor = 1.0  # Default
                    
                    # Combine factors
                    score = interaction_weight * recency_factor * similarity_factor
                    
                    # Update event score
                    event_id_str = str(event_id)
                    if event_id_str in event_scores:
                        event_scores[event_id_str] += score
                    else:
                        event_scores[event_id_str] = score
            
            # Normalize scores
            if event_scores:
                max_score = max(event_scores.values())
                if max_score > 0:
                    event_scores = {k: v/max_score for k, v in event_scores.items()}
        
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
        
        return event_scores
    
    def _store_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]]):
        """Store recommendations in the database for caching."""
        # Remove event data before storing
        for rec in recommendations:
            rec_copy = rec.copy()
            if "event" in rec_copy:
                del rec_copy["event"]
            
            # Add timestamp and user_id
            rec_copy["timestamp"] = datetime.now()
            rec_copy["user_id"] = user_id
            
            # Store in database
            self.recommendations_collection.insert_one(rec_copy)
    
    def _get_events_by_ids(self, event_ids: List[str]) -> List[Dict[str, Any]]:
        """Get full event details by IDs."""
        object_ids = [ObjectId(id) for id in event_ids]
        return list(self.events_collection.find({"_id": {"$in": object_ids}}))
    
    def _get_fallback_recommendations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get fallback recommendations when personalization is not possible."""
        # Get trending events as fallback
        now = datetime.now()
        trending_events = list(self.events_collection.find({
            "datetime.start": {"$gt": now}
        }).sort("views_count", -1).limit(limit))
        
        # Format as recommendations
        recommendations = []
        for event in trending_events:
            event_id = str(event["_id"])
            recommendations.append({
                "event_id": event_id,
                "score": 0.5,  # Default score
                "recommendation_type": "trending",
                "score_components": {"trending": 0.5},
                "event": event
            })
        
        return recommendations
    
    def invalidate_recommendations(self, user_id: str):
        """Invalidate cached recommendations for a user."""
        self.recommendations_collection.delete_many({"user_id": user_id})


# Create a singleton instance
recommendation_engine = RecommendationEngine()


def get_recommendations_for_user(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Public function to get recommendations for a user.
    
    Args:
        user_id: The ID of the user to get recommendations for
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommended events with scores
    """
    return recommendation_engine.generate_recommendations(user_id, limit)


def refresh_user_recommendations(user_id: str) -> bool:
    """
    Invalidate and regenerate recommendations for a user.
    
    Args:
        user_id: The ID of the user to refresh recommendations for
        
    Returns:
        True if successful, False otherwise
    """
    try:
        recommendation_engine.invalidate_recommendations(user_id)
        recommendation_engine.generate_recommendations(user_id)
        return True
    except Exception as e:
        logger.error(f"Error refreshing recommendations for user {user_id}: {str(e)}")
        return False


def update_recommendation_weights(new_weights: Dict[str, float]) -> bool:
    """
    Update the weights used in the recommendation algorithm.
    
    Args:
        new_weights: Dictionary of weight name to weight value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate weights
        required_weights = {
            "category_match", "location_proximity", "price_match", 
            "popularity", "recency", "collaborative"
        }
        
        if not all(weight in new_weights for weight in required_weights):
            logger.error(f"Missing required weights: {required_weights - new_weights.keys()}")
            return False
        
        # Update weights
        recommendation_engine.weights = new_weights
        return True
    except Exception as e:
        logger.error(f"Error updating recommendation weights: {str(e)}")
        return False