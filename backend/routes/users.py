"""
TufanTicket - User Routes
------------------------
This module implements FastAPI endpoints for user interactions and preferences,
including swipe logging, preference updates, and retrieving user interaction history.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path, status
from pydantic import BaseModel, Field, validator

# Import models and services
from ..models.user import UserModel
from ..models.interaction import (
    InteractionType, 
    SwipeDirection, 
    InteractionModel,
    get_user_interactions as fetch_user_interactions
)
from ..services.user_preferences import UserPreferenceService

# Create router
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


# Request/Response Models
class SwipeRequest(BaseModel):
    """Model for swipe interaction requests."""
    user_id: str
    event_id: str
    direction: SwipeDirection
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('direction')
    def validate_direction(cls, v):
        """Ensure direction is valid."""
        if v not in [SwipeDirection.LEFT, SwipeDirection.RIGHT]:
            raise ValueError(f"Invalid swipe direction: {v}")
        return v


class InteractionRequest(BaseModel):
    """Model for general interaction requests."""
    user_id: str
    event_id: str
    interaction_type: InteractionType
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InteractionResponse(BaseModel):
    """Model for interaction response."""
    interaction_id: str
    user_id: str
    event_id: str
    interaction_type: str
    timestamp: datetime


class UserPreferenceUpdate(BaseModel):
    """Model for user preference updates."""
    preferred_categories: Optional[List[str]] = None
    preferred_location: Optional[Dict[str, str]] = None
    price_range: Optional[Dict[str, float]] = None
    notification_preferences: Optional[Dict[str, bool]] = None


class UserInteractionStats(BaseModel):
    """Model for user interaction statistics."""
    interaction_counts: Dict[str, int]
    category_interests: Dict[str, int]
    total_interactions: int
    conversion_rate: float
    swipe_ratio: Dict[str, int]


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str


# Routes
@router.post(
    "/swipe",
    response_model=InteractionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "User or Event Not Found"}
    }
)
async def log_swipe(swipe_data: SwipeRequest):
    """
    Log a user's swipe on an event.
    
    - **user_id**: ID of the user
    - **event_id**: ID of the event
    - **direction**: Swipe direction (left/right)
    - **metadata**: Optional additional data about the swipe
    """
    try:
        interaction_id = UserPreferenceService.log_swipe(
            swipe_data.user_id,
            swipe_data.event_id,
            swipe_data.direction,
            swipe_data.metadata
        )
        
        # Get current timestamp
        timestamp = datetime.utcnow()
        
        interaction_type = (
            InteractionType.SWIPE_RIGHT if swipe_data.direction == SwipeDirection.RIGHT 
            else InteractionType.SWIPE_LEFT
        )
        
        # Return formatted response
        return {
            "interaction_id": interaction_id,
            "user_id": swipe_data.user_id,
            "event_id": swipe_data.event_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log swipe: {str(e)}"
        )


@router.post(
    "/interaction",
    response_model=InteractionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "User or Event Not Found"}
    }
)
async def log_interaction(interaction_data: InteractionRequest):
    """
    Log a user's interaction with an event.
    
    - **user_id**: ID of the user
    - **event_id**: ID of the event
    - **interaction_type**: Type of interaction (click, view, bookmark, etc.)
    - **metadata**: Optional additional data about the interaction
    """
    try:
        interaction_id = UserPreferenceService.log_interaction(
            interaction_data.user_id,
            interaction_data.event_id,
            interaction_data.interaction_type,
            interaction_data.metadata
        )
        
        # Get current timestamp
        timestamp = datetime.utcnow()
        
        # Return formatted response
        return {
            "interaction_id": interaction_id,
            "user_id": interaction_data.user_id,
            "event_id": interaction_data.event_id,
            "interaction_type": interaction_data.interaction_type,
            "timestamp": timestamp
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log interaction: {str(e)}"
        )


@router.get(
    "/{user_id}/interactions",
    response_model=List[Dict[str, Any]],
    responses={
        404: {"model": ErrorResponse, "description": "User Not Found"}
    }
)
async def get_user_interactions(
    user_id: str = Path(..., description="The ID of the user"),
    limit: int = Query(50, description="Maximum number of interactions to return"),
    interaction_type: Optional[str] = Query(None, description="Filter by interaction type")
):
    """
    Retrieve interaction history for a user.
    
    - **user_id**: ID of the user
    - **limit**: Maximum number of interactions to return (default: 50)
    - **interaction_type**: Optional filter by interaction type
    """
    try:
        # Check if user exists
        user_profile = UserPreferenceService.get_user_profile(user_id)
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
            
        # Get interactions
        interactions = fetch_user_interactions(user_id, limit)
        
        # Filter by interaction type if specified
        if interaction_type:
            try:
                interaction_enum = InteractionType(interaction_type)
                interactions = [
                    i for i in interactions 
                    if i.get("interaction_type") == interaction_enum
                ]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid interaction type: {interaction_type}"
                )
                
        return interactions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve interactions: {str(e)}"
        )


@router.put(
    "/{user_id}/preferences",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "User Not Found"}
    }
)
async def update_user_preferences(
    preferences: UserPreferenceUpdate,
    user_id: str = Path(..., description="The ID of the user")
):
    """
    Update user preferences.
    
    - **user_id**: ID of the user
    - **preferences**: Preference data to update
    """
    # Convert Pydantic model to dict
    pref_dict = preferences.dict(exclude_unset=True)
    
    # Check if user exists
    user_profile = UserPreferenceService.get_user_profile(user_id)
    if not user_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
        
    # Update preferences
    success = UserPreferenceService.update_user_preferences(user_id, pref_dict)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user preferences"
        )
        
    # Clear any cached data
    UserPreferenceService.invalidate_cache(user_id)
    
    return {"status": "success", "message": "User preferences updated successfully"}


@router.get(
    "/{user_id}/profile",
    responses={
        404: {"model": ErrorResponse, "description": "User Not Found"}
    }
)
async def get_user_profile(
    user_id: str = Path(..., description="The ID of the user")
):
    """
    Get a user's profile including preferences.
    
    - **user_id**: ID of the user
    """
    user_profile = UserPreferenceService.get_user_profile(user_id)
    
    if not user_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
        
    return user_profile


@router.get(
    "/{user_id}/stats",
    response_model=UserInteractionStats,
    responses={
        404: {"model": ErrorResponse, "description": "User Not Found"}
    }
)
async def get_user_stats(
    user_id: str = Path(..., description="The ID of the user")
):
    """
    Get statistics about a user's interactions.
    
    - **user_id**: ID of the user
    """
    # Check if user exists
    user_profile = UserPreferenceService.get_user_profile(user_id)
    if not user_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
        
    # Get interaction stats
    stats = UserPreferenceService.get_user_interaction_stats(user_id)
    
    return stats


@router.get(
    "/{user_id}/recommendation-data",
    responses={
        404: {"model": ErrorResponse, "description": "User Not Found"}
    }
)
async def get_recommendation_data(
    user_id: str = Path(..., description="The ID of the user")
):
    """
    Get user data needed for the recommendation engine.
    
    - **user_id**: ID of the user
    """
    try:
        data = UserPreferenceService.get_recommendation_data(user_id)
        return data
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recommendation data: {str(e)}"
        )