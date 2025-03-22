from datetime import datetime
from typing import List, Optional, Dict, Any

from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, validator, root_validator

# PyObjectId class for MongoDB ObjectId handling - similar to what's in event.py
class PyObjectId(ObjectId):
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


# Define available event categories - match with EventCategory in event.py
class UserEventCategory(str):
    MUSIC = "music"
    COMEDY = "comedy"
    THEATRE = "theatre"
    FOOD = "food"
    SPORTS = "sports"
    BUSINESS = "business"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    ARTS = "arts"
    FILM = "film"
    FESTIVAL = "festival"
    SPIRITUAL = "spiritual"
    HEALTH = "health"
    OTHER = "other"


# Model for tracking event interactions
class EventInteraction(BaseModel):
    event_id: str = Field(..., description="Unique ID of the event")
    interaction_type: str = Field(..., description="Type of interaction: view, like, bookmark, purchase")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the interaction occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional interaction details")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "event_id": "event123",
                "interaction_type": "purchase",
                "timestamp": "2023-06-15T10:30:00",
                "metadata": {
                    "ticket_tier": "premium",
                    "price_paid": 1500,
                    "quantity": 2
                }
            }
        }


# Model for tracking event bookings
class EventBooking(BaseModel):
    event_id: str = Field(..., description="Unique ID of the event")
    booking_id: str = Field(..., description="Unique booking reference ID")
    ticket_quantity: int = Field(..., ge=1, description="Number of tickets purchased")
    ticket_tier: Optional[str] = Field(None, description="Ticket tier/category purchased")
    total_amount: float = Field(..., ge=0, description="Total amount paid in INR")
    purchase_date: datetime = Field(default_factory=datetime.utcnow, description="When the booking was made")
    status: str = Field(default="confirmed", description="Booking status: confirmed, cancelled, pending")
    
    @validator('ticket_quantity')
    def validate_ticket_quantity(cls, v):
        if v <= 0:
            raise ValueError("Ticket quantity must be at least 1")
        return v
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "event_id": "event123",
                "booking_id": "BK12345",
                "ticket_quantity": 2,
                "ticket_tier": "VIP",
                "total_amount": 3000,
                "purchase_date": "2023-06-10T15:45:00",
                "status": "confirmed"
            }
        }


# Main User model
class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User's email address")
    name: str = Field(..., description="User's full name")
    phone: Optional[str] = Field(None, description="User's phone number")
    password_hash: str = Field(..., description="Hashed password (never store plaintext)")
    
    location: Dict[str, str] = Field(
        default_factory=lambda: {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
        description="User's location details"
    )
    
    preferred_categories: List[str] = Field(
        default_factory=list,
        description="User's preferred event categories"
    )
    
    preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "notification_email": True,
            "notification_sms": False,
            "max_price": 5000,
            "min_rating": 3.5,
            "preferred_venues": []
        },
        description="User's platform preferences"
    )
    
    interactions: List[EventInteraction] = Field(
        default_factory=list,
        description="Record of user interactions with events"
    )
    
    booking_history: List[EventBooking] = Field(
        default_factory=list,
        description="User's event booking history"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last account update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    is_active: bool = Field(default=True, description="Whether user account is active")
    is_verified: bool = Field(default=False, description="Whether user email is verified")
    
    # Custom validators
    @validator('preferred_categories')
    def validate_categories(cls, categories):
        valid_categories = [c for c in dir(UserEventCategory) if not c.startswith('_')]
        for category in categories:
            if category.lower() not in [c.lower() for c in valid_categories]:
                raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
        return categories
    
    @root_validator
    def update_timestamp(cls, values):
        values["updated_at"] = datetime.now()
        return values
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "user_id": "user123",
                "email": "user@example.com",
                "name": "Raj Sharma",
                "phone": "9876543210",
                "password_hash": "hashed_password_string",
                "location": {
                    "city": "Bangalore",
                    "state": "Karnataka",
                    "country": "India"
                },
                "preferred_categories": ["music", "comedy", "food"],
                "preferences": {
                    "notification_email": True,
                    "notification_sms": False,
                    "max_price": 3000,
                    "min_rating": 4.0,
                    "preferred_venues": ["Phoenix Marketcity", "Indiranagar Social"]
                }
            }
        }


# Model for creating a new user (subset of UserModel)
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    phone: Optional[str] = None
    password: str  # Plain password (will be hashed before storage)
    location: Dict[str, str] = {"city": "Mumbai", "state": "Maharashtra", "country": "India"}
    preferred_categories: List[str] = []
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "new.user@example.com",
                "name": "Priya Patel",
                "phone": "9876543210",
                "password": "SecurePass123",
                "location": {
                    "city": "Delhi",
                    "state": "Delhi",
                    "country": "India"
                },
                "preferred_categories": ["music", "food", "technology"]
            }
        }


# Model for updating user information
class UserUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    preferred_categories: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None
    
    @validator('preferred_categories')
    def validate_categories(cls, categories):
        if categories is None:
            return None
            
        valid_categories = [c for c in dir(UserEventCategory) if not c.startswith('_')]
        for category in categories:
            if category.lower() not in [c.lower() for c in valid_categories]:
                raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
        return categories
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Priya Singh",
                "location": {
                    "city": "Pune",
                    "state": "Maharashtra",
                    "country": "India"
                },
                "preferred_categories": ["theatre", "business"]
            }
        }


# User response model (for API responses)
class UserResponse(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str
    email: EmailStr
    name: str
    phone: Optional[str]
    location: Dict[str, str]
    preferred_categories: List[str]
    preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    is_verified: bool
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Database operations for User model
async def get_user_by_id(user_id: str, db):
    """Retrieve a user by user_id"""
    user = await db.users.find_one({"user_id": user_id})
    if user:
        return user
    return None


async def get_user_by_email(email: str, db):
    """Retrieve a user by email"""
    user = await db.users.find_one({"email": email})
    if user:
        return user
    return None


async def create_user(user_data: dict, db):
    """Create a new user in the database"""
    user = await db.users.insert_one(user_data)
    return user


async def update_user(user_id: str, update_data: dict, db):
    """Update a user's information"""
    # Set updated_at timestamp
    update_data["updated_at"] = datetime.now()
    
    user = await db.users.update_one(
        {"user_id": user_id},
        {"$set": update_data}
    )
    return user


async def add_event_interaction(user_id: str, interaction: dict, db):
    """Add an event interaction to user's history"""
    result = await db.users.update_one(
        {"user_id": user_id},
        {"$push": {"interactions": interaction}}
    )
    return result


async def add_event_booking(user_id: str, booking: dict, db):
    """Add an event booking to user's history"""
    result = await db.users.update_one(
        {"user_id": user_id},
        {"$push": {"booking_history": booking}}
    )
    return result


async def get_user_interactions(user_id: str, db, limit: int = 50):
    """Get a user's recent event interactions"""
    user = await db.users.find_one(
        {"user_id": user_id},
        {"interactions": {"$slice": -limit}}
    )
    if user and "interactions" in user:
        return user["interactions"]
    return []


async def get_user_bookings(user_id: str, db, limit: int = 20):
    """Get a user's recent event bookings"""
    user = await db.users.find_one(
        {"user_id": user_id},
        {"booking_history": {"$slice": -limit}}
    )
    if user and "booking_history" in user:
        return user["booking_history"]
    return []


# Initialize MongoDB indexes for User model
async def init_user_indexes(db):
    """Initialize MongoDB indexes for User collection"""
    # Create indexes for frequently queried fields
    await db.users.create_index("user_id", unique=True)
    await db.users.create_index("email", unique=True)
    await db.users.create_index("location.city")
    await db.users.create_index("preferred_categories")
    
    # Index for full-text search on name
    await db.users.create_index([("name", "text")])
    
    # Compound index for analytics queries
    await db.users.create_index([
        ("location.city", 1),
        ("preferred_categories", 1)
    ])
    
    print("User collection indexes initialized successfully")