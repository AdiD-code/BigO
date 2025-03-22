"""
API endpoints for accessing organizer analytics.
Provides event trend analysis, pricing strategy suggestions, and anomaly detections.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import pymongo
from bson import ObjectId

from backend.models.event import EventCategory
from backend.services.organizer_analytics import forecast_sales as forecast_ticket_sales
from backend.services.organizer_analytics import (
    analyze_event_trends as analyze_trends,
    analyze_review_sentiment as analyze_sentiment,
    detect_sales_anomalies as detect_anomalies,
    suggest_optimal_pricing as optimize_pricing,
    get_event_performance_metrics,
    get_conversion_funnels
)
from backend.config import ADVANCED_ANALYTICS_ENABLED


# ----- Pydantic Models ----- #
class TimeRange(BaseModel):
    start_date: datetime = Field(..., description="Start date for analytics data")
    end_date: datetime = Field(..., description="End date for analytics data")
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class OrganizerQuery(BaseModel):
    organizer_id: str = Field(..., description="ID of the event organizer")
    event_id: Optional[str] = Field(None, description="Specific event ID (optional)")
    time_range: TimeRange = Field(default_factory=lambda: TimeRange(
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now() + timedelta(days=90)
    ))
    categories: Optional[List[EventCategory]] = Field(None, description="Filter by event categories")
    city: Optional[str] = Field(None, description="Filter by city")


class SalesForecast(BaseModel):
    event_id: str
    event_name: str
    predicted_sales: Dict[str, int] = Field(..., description="Daily forecasted sales for the next 30 days")
    confidence_interval: Dict[str, List[int]] = Field(..., description="95% confidence intervals for predictions")
    total_predicted: int = Field(..., description="Total predicted sales")
    factors: Dict[str, float] = Field(..., description="Factors influencing the prediction")


class TrendAnalysis(BaseModel):
    popular_categories: List[Dict[str, Union[str, int, float]]]
    seasonal_trends: Dict[str, List[Dict[str, Union[str, float]]]]
    category_growth: Dict[str, float]
    geographic_hotspots: List[Dict[str, Union[str, int]]]
    peak_booking_times: Dict[str, Dict[str, int]]


class SentimentAnalysis(BaseModel):
    overall_sentiment: float
    aspect_sentiments: Dict[str, float]
    frequent_positive_terms: List[str]
    frequent_negative_terms: List[str]
    sentiment_trend: Dict[str, float]


class AnomalyDetection(BaseModel):
    anomalies: List[Dict[str, Union[str, datetime, float, str]]]
    analysis: str


class PricingRecommendation(BaseModel):
    event_id: str
    event_name: str
    current_price: float
    recommended_price: float
    price_range: List[float]
    conversion_impact: Dict[str, float]
    similar_events: List[Dict[str, Union[str, float]]]


class ConversionFunnel(BaseModel):
    stage: str
    count: int
    conversion_rate: float


class EventPerformance(BaseModel):
    event_id: str
    event_name: str
    views: int
    interactions: int
    bookings: int
    conversion_rate: float


class OrganizerInsightsResponse(BaseModel):
    sales_forecast: Optional[List[SalesForecast]] = None
    trend_analysis: Optional[TrendAnalysis] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    anomaly_detection: Optional[AnomalyDetection] = None
    pricing_recommendations: Optional[List[PricingRecommendation]] = None
    event_performance: Optional[List[EventPerformance]] = None
    conversion_funnels: Optional[List[ConversionFunnel]] = None


# ----- Router Setup ----- #
router = APIRouter(
    prefix="/organizer",
    tags=["Organizer Analytics"],
    responses={
        404: {"description": "Not found"},
        403: {"description": "Forbidden - insufficient permissions"},
        500: {"description": "Internal server error"}
    }
)


# ----- Helper Functions ----- #
def validate_organizer_access(organizer_id: str, event_id: Optional[str] = None) -> bool:
    """Validates that the organizer has access to the requested event data."""
    # In a real implementation, this would check user permissions
    # For now, we'll just return True
    return True


# ----- Endpoints ----- #
@router.get(
    "/insights",
    response_model=OrganizerInsightsResponse,
    summary="Get comprehensive insights for event organizers",
    description="Returns event trend analysis, pricing strategy suggestions, and anomaly detections."
)
async def get_organizer_insights(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)"),
    start_date: Optional[datetime] = Query(
        datetime.now() - timedelta(days=90),
        description="Start date for analytics"
    ),
    end_date: Optional[datetime] = Query(
        datetime.now() + timedelta(days=90),
        description="End date for analytics"
    ),
    category: Optional[EventCategory] = Query(None, description="Filter by event category")
):
    # Validate organizer access
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    # Create query parameters
    query = OrganizerQuery(
        organizer_id=organizer_id,
        event_id=event_id,
        time_range=TimeRange(start_date=start_date, end_date=end_date),
        categories=[category] if category else None
    )
    
    try:
        # Get all insights data
        insights = OrganizerInsightsResponse(
            sales_forecast=get_sales_forecast(
                organizer_id=query.organizer_id,
                event_id=query.event_id,
                start_date=query.time_range.start_date,
                end_date=query.time_range.end_date,
                categories=query.categories
            ) if ADVANCED_ANALYTICS_ENABLED else None,
            
            trend_analysis=analyze_trends(
                organizer_id=query.organizer_id,
                start_date=query.time_range.start_date,
                end_date=query.time_range.end_date,
                categories=query.categories,
                city=query.city
            ),
            
            sentiment_analysis=analyze_sentiment(
                organizer_id=query.organizer_id,
                event_id=query.event_id
            ) if ADVANCED_ANALYTICS_ENABLED else None,
            
            anomaly_detection=detect_anomalies(
                organizer_id=query.organizer_id,
                event_id=query.event_id,
                start_date=query.time_range.start_date,
                end_date=query.time_range.end_date
            ) if ADVANCED_ANALYTICS_ENABLED else None,
            
            pricing_recommendations=optimize_pricing(
                organizer_id=query.organizer_id,
                event_id=query.event_id,
                categories=query.categories,
                city=query.city
            ),
            
            event_performance=get_event_performance_metrics(
                organizer_id=query.organizer_id,
                event_id=query.event_id,
                start_date=query.time_range.start_date,
                end_date=query.time_range.end_date
            ),
            
            conversion_funnels=get_conversion_funnels(
                organizer_id=query.organizer_id,
                event_id=query.event_id,
                start_date=query.time_range.start_date,
                end_date=query.time_range.end_date
            )
        )
        
        return insights
    
    except Exception as e:
        # Log the exception details
        print(f"Error generating insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate insights: {str(e)}"
        )


@router.get(
    "/sales-forecast",
    response_model=List[SalesForecast],
    summary="Get sales forecast for events",
    description="Returns forecasted ticket sales for upcoming events."
)
async def get_event_sales_forecast(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)"),
    days_ahead: int = Query(30, description="Number of days to forecast")
):
    if not ADVANCED_ANALYTICS_ENABLED:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Advanced analytics features are not enabled"}
        )
    
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        forecast_data = get_sales_forecast(
            organizer_id=organizer_id,
            event_id=event_id,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=days_ahead)
        )
        
        return forecast_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error generating sales forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sales forecast: {str(e)}"
        )


@router.get(
    "/trends",
    response_model=TrendAnalysis,
    summary="Get event trend analysis",
    description="Returns analysis of popular categories, seasonal trends, and geographic hotspots."
)
async def get_trend_analysis(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    start_date: Optional[datetime] = Query(
        datetime.now() - timedelta(days=90),
        description="Start date for trend analysis"
    ),
    end_date: Optional[datetime] = Query(
        datetime.now() + timedelta(days=90),
        description="End date for trend analysis"
    ),
    category: Optional[EventCategory] = Query(None, description="Filter by event category"),
    city: Optional[str] = Query(None, description="Filter by city")
):
    if not validate_organizer_access(organizer_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        categories = [category] if category else None
        
        trend_data = analyze_trends(
            organizer_id=organizer_id,
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            city=city
        )
        
        return trend_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error generating trend analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate trend analysis: {str(e)}"
        )


@router.get(
    "/pricing",
    response_model=List[PricingRecommendation],
    summary="Get pricing recommendations",
    description="Returns optimal pricing recommendations based on similar events and conversion rates."
)
async def get_pricing_recommendations(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)"),
    category: Optional[EventCategory] = Query(None, description="Filter by event category"),
    city: Optional[str] = Query(None, description="Filter by city")
):
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        categories = [category] if category else None
        
        pricing_data = optimize_pricing(
            organizer_id=organizer_id,
            event_id=event_id,
            categories=categories,
            city=city
        )
        
        return pricing_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error generating pricing recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate pricing recommendations: {str(e)}"
        )


@router.get(
    "/performance",
    response_model=List[EventPerformance],
    summary="Get event performance metrics",
    description="Returns performance metrics for events including views, interactions, and conversion rates."
)
async def get_performance_metrics(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)"),
    start_date: Optional[datetime] = Query(
        datetime.now() - timedelta(days=30),
        description="Start date for performance metrics"
    ),
    end_date: Optional[datetime] = Query(
        datetime.now(),
        description="End date for performance metrics"
    )
):
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        performance_data = get_event_performance_metrics(
            organizer_id=organizer_id,
            event_id=event_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return performance_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error retrieving event performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve event performance metrics: {str(e)}"
        )


@router.get(
    "/sentiment",
    response_model=SentimentAnalysis,
    summary="Get sentiment analysis",
    description="Returns sentiment analysis for event reviews including aspect-based analysis."
)
async def get_sentiment_analysis(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)")
):
    if not ADVANCED_ANALYTICS_ENABLED:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Advanced analytics features are not enabled"}
        )
    
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        sentiment_data = analyze_sentiment(
            organizer_id=organizer_id,
            event_id=event_id
        )
        
        return sentiment_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error generating sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sentiment analysis: {str(e)}"
        )


@router.get(
    "/anomalies",
    response_model=AnomalyDetection,
    summary="Get anomaly detection",
    description="Returns detected anomalies in ticket sales and user interactions."
)
async def get_anomaly_detection(
    organizer_id: str = Query(..., description="ID of the event organizer"),
    event_id: Optional[str] = Query(None, description="Specific event ID (optional)"),
    start_date: Optional[datetime] = Query(
        datetime.now() - timedelta(days=30),
        description="Start date for anomaly detection"
    ),
    end_date: Optional[datetime] = Query(
        datetime.now(),
        description="End date for anomaly detection"
    )
):
    if not ADVANCED_ANALYTICS_ENABLED:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Advanced analytics features are not enabled"}
        )
    
    if not validate_organizer_access(organizer_id, event_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this data"
        )
    
    try:
        anomaly_data = detect_anomalies(
            organizer_id=organizer_id,
            event_id=event_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return anomaly_data
    
    except Exception as e:
        # Log the exception details
        print(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect anomalies: {str(e)}"
        )