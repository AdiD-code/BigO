"""
TufanTicket Organizer Analytics Service
======================================
This module provides AI-powered event analytics for event organizers.
It uses machine learning to predict trends, forecast ticket sales,
perform sentiment analysis on reviews, and detect anomalies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# ML imports
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import IsolationForest

# Local imports
from ..models.event import EventModel
from ..models.user import UserModel, EventBooking
from ..models.interaction import InteractionModel, InteractionType
from ..config import (
    MONGODB_URI, 
    MONGODB_DB_NAME, 
    ANALYTICS_CACHE_EXPIRY,
    SALES_FORECAST_WINDOW_DAYS,
    TREND_ANALYSIS_MIN_SAMPLE,
    ANOMALY_DETECTION_CONTAMINATION
)

# MongoDB connection
import pymongo
from bson import ObjectId

# Initialize MongoDB client
client = pymongo.MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize sentiment analyzer: {e}")
    sentiment_analyzer = None

class OrganizerAnalytics:
    """
    Provides analytics and predictions for event organizers
    """
    
    def __init__(self):
        self.events_collection = db.events
        self.users_collection = db.users
        self.interactions_collection = db.interactions
        self.bookings_collection = db.bookings
        self.reviews_collection = db.reviews
        
        # Initialize ML models
        self.sales_forecast_model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=ANOMALY_DETECTION_CONTAMINATION,
            random_state=42
        )
    
    # ===== DATA COLLECTION & PREPARATION =====
    
    def _get_event_data(self, event_id: str) -> Dict:
        """
        Fetch the event data by ID
        """
        return self.events_collection.find_one({"_id": ObjectId(event_id)})
    
    def _get_organizer_events(self, organizer_id: str) -> List[Dict]:
        """
        Fetch all events by a specific organizer
        """
        return list(self.events_collection.find({"organizer.id": organizer_id}))
    
    def _get_event_bookings(self, event_id: str) -> List[Dict]:
        """
        Fetch all bookings for a specific event
        """
        return list(self.bookings_collection.find({"event_id": event_id}))
    
    def _get_event_interactions(self, event_id: str) -> List[Dict]:
        """
        Fetch all user interactions for a specific event
        """
        return list(self.interactions_collection.find({"event_id": event_id}))
    
    def _get_event_reviews(self, event_id: str) -> List[Dict]:
        """
        Fetch all reviews for a specific event
        """
        return list(self.reviews_collection.find({"event_id": event_id}))
    
    def get_event_performance_metrics(event_id, db):
        """
        Retrieve performance metrics for a given event from the database.
        
        Metrics include:
        - Total ticket sales
        - Attendance rate
        - Revenue generated
        - User engagement score
        """
        event = db.events.find_one({"event_id": event_id})
        if not event:
            return {"error": "Event not found"}
        
        return {
            "event_id": event_id,
            "ticket_sales": event.get("ticket_sales", 0),
            "attendance_rate": event.get("attendance_rate", 0),
            "revenue": event.get("revenue", 0),
            "engagement_score": event.get("engagement_score", 0)
        }


    def get_conversion_funnels(event_id, db):
        """
        Analyze the conversion funnel for an event.

        Stages:
        - Event Views
        - Tickets Added to Cart
        - Ticket Purchases
        - Conversion Rate Calculation
        """
        event = db.events.find_one({"event_id": event_id})
        if not event:
            return {"error": "Event not found"}
        
        views = event.get("views", 0)
        cart_adds = event.get("cart_adds", 0)
        purchases = event.get("purchases", 0)
        
        conversion_rate = (purchases / views) * 100 if views else 0

        return {
            "event_id": event_id,
            "views": views,
            "added_to_cart": cart_adds,
            "purchased": purchases,
            "conversion_rate": conversion_rate
        }

    
    def _prepare_sales_data(self, event_id: str = None, organizer_id: str = None) -> pd.DataFrame:
        """
        Prepare historical sales data for forecasting
        
        Args:
            event_id: Specific event ID or None for all events
            organizer_id: Specific organizer ID or None for all organizers
            
        Returns:
            DataFrame with historical sales data
        """
        query = {}
        if event_id:
            query["event_id"] = event_id
        elif organizer_id:
            # Get all events by this organizer
            organizer_events = self._get_organizer_events(organizer_id)
            event_ids = [str(event["_id"]) for event in organizer_events]
            query["event_id"] = {"$in": event_ids}
        
        # Get all bookings that match the query
        bookings = list(self.bookings_collection.find(query))
        
        if not bookings:
            logger.warning(f"No booking data found for query: {query}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(bookings)
        
        # Convert booking_time to datetime
        df['booking_time'] = pd.to_datetime(df['booking_time'])
        
        # Group by date and aggregate
        daily_sales = df.groupby(df['booking_time'].dt.date).agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            '_id': 'count'  # Count of transactions
        }).rename(columns={'_id': 'transaction_count'})
        
        # Add time-based features
        daily_sales = daily_sales.reset_index()
        daily_sales['booking_time'] = pd.to_datetime(daily_sales['booking_time'])
        daily_sales['day_of_week'] = daily_sales['booking_time'].dt.dayofweek
        daily_sales['month'] = daily_sales['booking_time'].dt.month
        daily_sales['is_weekend'] = daily_sales['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Sort by date
        daily_sales = daily_sales.sort_values('booking_time')
        
        # Fill any gaps in the date range
        if not daily_sales.empty:
            date_range = pd.date_range(
                start=daily_sales['booking_time'].min(),
                end=daily_sales['booking_time'].max()
            )
            daily_sales = daily_sales.set_index('booking_time').reindex(date_range).fillna(0).reset_index()
            daily_sales = daily_sales.rename(columns={'index': 'booking_time'})
        
        return daily_sales
    
    # ===== SALES FORECASTING =====

    def forecast_sales(self, event_id, db):
        """
        Predict future ticket sales for an event based on historical data.
        """
        event = db.events.find_one({"event_id": event_id})
        if not event:
            return {"error": "Event not found"}

        return {
            "event_id": event_id,
            "predicted_sales": event.get("ticket_sales", 0) * 1.1  # Example: 10% growth prediction
        }
    
    def _train_forecast_model(self, sales_data: pd.DataFrame) -> None:
        """
        Train XGBoost model for sales forecasting
        """
        if sales_data.empty or len(sales_data) < TREND_ANALYSIS_MIN_SAMPLE:
            logger.warning("Insufficient data to train sales forecast model")
            return
        
        try:
            # Features for prediction
            features = ['day_of_week', 'month', 'is_weekend']
            
            # Add lag features
            for lag in [1, 2, 3, 7]:  # Previous days, week
                sales_data[f'quantity_lag_{lag}'] = sales_data['quantity'].shift(lag)
                features.append(f'quantity_lag_{lag}')
            
            # Rolling averages
            for window in [3, 7, 14]:
                sales_data[f'quantity_roll_{window}'] = sales_data['quantity'].rolling(window=window).mean()
                features.append(f'quantity_roll_{window}')
            
            # Drop rows with NaN (from lag features)
            sales_data = sales_data.dropna()
            
            if len(sales_data) < 10:  # Need minimum data after creating features
                logger.warning("Insufficient data after feature engineering")
                return
            
            # Target variables
            X = sales_data[features]
            y = sales_data['quantity']
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train XGBoost model
            self.sales_forecast_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            self.sales_forecast_model.fit(X_train, y_train)
            
            # Log model performance
            train_score = self.sales_forecast_model.score(X_train, y_train)
            test_score = self.sales_forecast_model.score(X_test, y_test)
            logger.info(f"Sales forecast model trained. Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error training sales forecast model: {e}")
    
    def forecast_ticket_sales(self, event_id: str = None, organizer_id: str = None, days_ahead: int = SALES_FORECAST_WINDOW_DAYS) -> Dict:
        """
        Forecast ticket sales for the next specified days
        
        Args:
            event_id: Specific event ID or None
            organizer_id: Specific organizer ID or None
            days_ahead: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        sales_data = self._prepare_sales_data(event_id, organizer_id)
        
        if sales_data.empty or len(sales_data) < TREND_ANALYSIS_MIN_SAMPLE:
            return {
                "status": "insufficient_data",
                "message": "Not enough historical data for forecasting",
                "forecast": []
            }
        
        # Train model if not already trained
        if self.sales_forecast_model is None:
            self._train_forecast_model(sales_data)
        
        # If training failed
        if self.sales_forecast_model is None:
            return {
                "status": "model_error",
                "message": "Could not train forecast model",
                "forecast": []
            }
        
        try:
            # Prepare data for prediction
            features = ['day_of_week', 'month', 'is_weekend']
            lag_features = [f'quantity_lag_{lag}' for lag in [1, 2, 3, 7]]
            roll_features = [f'quantity_roll_{window}' for window in [3, 7, 14]]
            
            # Last known values for lag features
            last_quantities = sales_data['quantity'].tail(7).tolist()
            
            # Generate dates for prediction
            last_date = sales_data['booking_time'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
            
            # Create prediction dataframe
            pred_data = pd.DataFrame({
                'booking_time': future_dates,
                'day_of_week': [d.dayofweek for d in future_dates],
                'month': [d.month for d in future_dates],
                'is_weekend': [1 if d.dayofweek >= 5 else 0 for d in future_dates]
            })
            
            # Initialize with zeros
            for feature in lag_features + roll_features:
                pred_data[feature] = 0
            
            # Make predictions day by day
            predictions = []
            
            for i in range(days_ahead):
                day_data = pred_data.iloc[i:i+1].copy()
                
                # Update lag features
                for j, lag in enumerate([1, 2, 3, 7]):
                    if i >= lag:
                        # Use previous prediction
                        day_data[f'quantity_lag_{lag}'] = predictions[i-lag]
                    elif len(last_quantities) > lag-1:
                        # Use historical data
                        day_data[f'quantity_lag_{lag}'] = last_quantities[-(lag-i)]
                
                # Update rolling averages
                for window in [3, 7, 14]:
                    values = []
                    for w in range(window):
                        if i >= w:
                            values.append(predictions[i-w])
                        elif len(last_quantities) > w-i:
                            values.append(last_quantities[-(w-i+1)])
                    
                    if values:
                        day_data[f'quantity_roll_{window}'] = sum(values) / len(values)
                
                # Scale features
                features_to_predict = features + lag_features + roll_features
                X_pred = self.scaler.transform(day_data[features_to_predict])
                
                # Make prediction
                prediction = max(0, round(float(self.sales_forecast_model.predict(X_pred)[0])))
                predictions.append(prediction)
            
            # Format results
            forecast_result = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "forecasted_tickets": int(pred),
                    "confidence": 0.8  # Fixed value, could be improved
                }
                for date, pred in zip(future_dates, predictions)
            ]
            
            return {
                "status": "success",
                "message": f"Successfully forecasted sales for next {days_ahead} days",
                "forecast": forecast_result,
                "total_forecasted_tickets": sum(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting ticket sales: {e}")
            return {
                "status": "error",
                "message": f"Error generating forecast: {str(e)}",
                "forecast": []
            }
    
    # ===== TREND ANALYSIS =====
    
    def analyze_event_trends(self, organizer_id: str) -> Dict:
        """
        Analyze event trends for a specific organizer
        
        Args:
            organizer_id: Organizer ID to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        # Get all events by this organizer
        organizer_events = self._get_organizer_events(organizer_id)
        
        if not organizer_events:
            return {
                "status": "no_data",
                "message": "No events found for this organizer",
                "trends": {}
            }
        
        # Convert to DataFrame
        events_df = pd.DataFrame(organizer_events)
        
        # Ensure datetime fields are properly formatted
        if 'datetime' in events_df.columns:
            if 'start' in events_df['datetime'].iloc[0]:
                events_df['start_date'] = pd.to_datetime(
                    events_df['datetime'].apply(lambda x: x.get('start', None))
                )
        
        # Get all interactions for these events
        event_ids = [str(event["_id"]) for event in organizer_events]
        interactions = list(self.interactions_collection.find({
            "event_id": {"$in": event_ids}
        }))
        
        if not interactions:
            return {
                "status": "no_interactions",
                "message": "No interaction data found for events",
                "trends": {}
            }
        
        # Convert to DataFrame
        interactions_df = pd.DataFrame(interactions)
        
        # Analyze category popularity
        category_trends = {}
        if 'category' in events_df.columns:
            category_counts = events_df['category'].value_counts().to_dict()
            category_interactions = {}
            
            for event_id, category in zip(events_df['_id'], events_df['category']):
                event_interactions = interactions_df[interactions_df['event_id'] == str(event_id)]
                if category in category_interactions:
                    category_interactions[category] += len(event_interactions)
                else:
                    category_interactions[category] = len(event_interactions)
            
            # Calculate engagement rate by category
            for category in category_counts:
                if category in category_interactions:
                    engagement = category_interactions[category] / category_counts[category]
                else:
                    engagement = 0
                
                category_trends[category] = {
                    "event_count": category_counts[category],
                    "interaction_count": category_interactions.get(category, 0),
                    "engagement_rate": engagement
                }
        
        # Analyze seasonal trends if date data is available
        seasonal_trends = {}
        if 'start_date' in events_df.columns and not events_df['start_date'].isna().all():
            events_df['month'] = events_df['start_date'].dt.month
            events_df['day_of_week'] = events_df['start_date'].dt.dayofweek
            
            # Monthly trends
            monthly_events = events_df['month'].value_counts().sort_index().to_dict()
            seasonal_trends["monthly"] = {
                f"month_{month}": count for month, count in monthly_events.items()
            }
            
            # Day of week trends
            dow_events = events_df['day_of_week'].value_counts().sort_index().to_dict()
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            seasonal_trends["day_of_week"] = {
                day_names[dow]: count for dow, count in dow_events.items() if dow < len(day_names)
            }
        
        # Analyze booking conversion rates
        bookings = list(self.bookings_collection.find({
            "event_id": {"$in": event_ids}
        }))
        
        conversion_metrics = {}
        if bookings:
            bookings_df = pd.DataFrame(bookings)
            
            # Group interactions by event and type
            interaction_counts = interactions_df.groupby(['event_id', 'interaction_type']).size().reset_index(name='count')
            
            # View to booking conversion
            views = interaction_counts[interaction_counts['interaction_type'] == 'view'].set_index('event_id')['count'].to_dict()
            booking_counts = bookings_df.groupby('event_id').size().to_dict()
            
            for event_id in views:
                if event_id in booking_counts:
                    conversion = (booking_counts[event_id] / views[event_id]) * 100
                else:
                    conversion = 0
                
                conversion_metrics[event_id] = {
                    "view_count": views[event_id],
                    "booking_count": booking_counts.get(event_id, 0),
                    "conversion_rate": conversion
                }
        
        return {
            "status": "success",
            "message": "Successfully analyzed event trends",
            "trends": {
                "category_trends": category_trends,
                "seasonal_trends": seasonal_trends,
                "conversion_metrics": conversion_metrics,
                "total_events": len(organizer_events),
                "total_interactions": len(interactions)
            }
        }
    
    # ===== SENTIMENT ANALYSIS =====
    
    def analyze_review_sentiment(self, event_id: str = None, organizer_id: str = None) -> Dict:
        """
        Perform sentiment analysis on event reviews
        
        Args:
            event_id: Specific event ID or None
            organizer_id: Specific organizer ID or None
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not sentiment_analyzer:
            return {
                "status": "error",
                "message": "Sentiment analyzer not available",
                "sentiment": {}
            }
        
        # Get reviews
        query = {}
        if event_id:
            query["event_id"] = event_id
        elif organizer_id:
            # Get all events by this organizer
            organizer_events = self._get_organizer_events(organizer_id)
            event_ids = [str(event["_id"]) for event in organizer_events]
            query["event_id"] = {"$in": event_ids}
        
        reviews = list(self.reviews_collection.find(query))
        
        if not reviews:
            return {
                "status": "no_reviews",
                "message": "No reviews found",
                "sentiment": {}
            }
        
        # Analyze sentiment for each review
        sentiment_scores = []
        topic_sentiments = {}
        
        for review in reviews:
            if 'text' in review and review['text']:
                # Overall sentiment
                sentiment = sentiment_analyzer.polarity_scores(review['text'])
                sentiment_scores.append(sentiment)
                
                # Extract topics/aspects (simplified)
                topics = self._extract_review_topics(review['text'])
                
                # Analyze sentiment for each topic
                for topic in topics:
                    if topic not in topic_sentiments:
                        topic_sentiments[topic] = []
                    
                    topic_sentiments[topic].append(sentiment['compound'])
        
        if not sentiment_scores:
            return {
                "status": "no_content",
                "message": "No content to analyze in reviews",
                "sentiment": {}
            }
        
        # Calculate average scores
        avg_sentiment = {
            "compound": sum(s['compound'] for s in sentiment_scores) / len(sentiment_scores),
            "positive": sum(s['pos'] for s in sentiment_scores) / len(sentiment_scores),
            "neutral": sum(s['neu'] for s in sentiment_scores) / len(sentiment_scores),
            "negative": sum(s['neg'] for s in sentiment_scores) / len(sentiment_scores)
        }
        
        # Calculate sentiment by topic
        topic_analysis = {}
        for topic, scores in topic_sentiments.items():
            avg_score = sum(scores) / len(scores)
            topic_analysis[topic] = {
                "avg_sentiment": avg_score,
                "sentiment_label": "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral",
                "mention_count": len(scores)
            }
        
        # Determine overall sentiment label
        sentiment_label = "positive"
        if avg_sentiment["compound"] < -0.05:
            sentiment_label = "negative"
        elif avg_sentiment["compound"] <= 0.05:
            sentiment_label = "neutral"
        
        return {
            "status": "success",
            "message": "Successfully analyzed review sentiment",
            "sentiment": {
                "overall_sentiment": {
                    "score": avg_sentiment["compound"],
                    "label": sentiment_label,
                    "positive_score": avg_sentiment["positive"],
                    "neutral_score": avg_sentiment["neutral"],
                    "negative_score": avg_sentiment["negative"]
                },
                "topic_sentiment": topic_analysis,
                "review_count": len(reviews)
            }
        }
    
    def _extract_review_topics(self, text: str) -> List[str]:
        """
        Extract topics/aspects from a review text (simplified)
        
        Note: In a production system, this would use more sophisticated NLP techniques
        """
        # Simplified topic extraction
        topics = set()
        
        # Check for common event aspects
        topic_keywords = {
            "venue": ["venue", "location", "place", "arena", "stadium", "hall", "space"],
            "sound": ["sound", "audio", "acoustics", "music", "volume", "noise"],
            "visuals": ["visuals", "lights", "lighting", "stage", "screen", "effects", "view"],
            "price": ["price", "cost", "expensive", "cheap", "worth", "value", "money", "ticket"],
            "staff": ["staff", "service", "security", "employee", "worker", "organizer"],
            "food": ["food", "drinks", "beverage", "catering", "refreshment", "snack"],
            "crowd": ["crowd", "audience", "people", "attendees", "busy", "packed", "empty"],
            "performer": ["performer", "artist", "band", "DJ", "actor", "speaker", "presenter"],
            "schedule": ["schedule", "time", "duration", "length", "long", "short", "late", "early"],
            "comfort": ["comfort", "comfortable", "seat", "seating", "chair", "standing"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.add(topic)
                    break
        
        return list(topics)
    
    # ===== ANOMALY DETECTION =====
    
    def detect_sales_anomalies(self, event_id: str = None, organizer_id: str = None) -> Dict:
        """
        Detect anomalies in ticket sales
        
        Args:
            event_id: Specific event ID or None
            organizer_id: Specific organizer ID or None
            
        Returns:
            Dictionary with anomaly detection results
        """
        sales_data = self._prepare_sales_data(event_id, organizer_id)
        
        if sales_data.empty or len(sales_data) < TREND_ANALYSIS_MIN_SAMPLE:
            return {
                "status": "insufficient_data",
                "message": "Not enough data for anomaly detection",
                "anomalies": []
            }
        
        try:
            # Features for anomaly detection
            features = ['quantity', 'transaction_count']
            if 'total_amount' in sales_data.columns:
                features.append('total_amount')
            
            # Add rolling statistics
            for col in ['quantity', 'transaction_count']:
                if col in sales_data.columns:
                    sales_data[f'{col}_rolling_mean'] = sales_data[col].rolling(window=7, min_periods=1).mean()
                    sales_data[f'{col}_rolling_std'] = sales_data[col].rolling(window=7, min_periods=1).std().fillna(0)
                    features.extend([f'{col}_rolling_mean', f'{col}_rolling_std'])
            
            # Extract features for anomaly detection
            X = sales_data[features].copy()
            
            # Fit anomaly detector
            self.anomaly_detector.fit(X)
            
            # Predict anomalies
            anomaly_scores = self.anomaly_detector.decision_function(X)
            anomaly_predictions = self.anomaly_detector.predict(X)
            
            # -1 indicates anomaly
            anomaly_indices = np.where(anomaly_predictions == -1)[0]
            
            # Format results
            anomalies = []
            for idx in anomaly_indices:
                anomalies.append({
                    "date": sales_data.iloc[idx]['booking_time'].strftime("%Y-%m-%d"),
                    "tickets_sold": int(sales_data.iloc[idx]['quantity']),
                    "transactions": int(sales_data.iloc[idx]['transaction_count']),
                    "anomaly_score": float(anomaly_scores[idx]),
                    "is_positive": sales_data.iloc[idx]['quantity'] > sales_data.iloc[idx]['quantity_rolling_mean'] if 'quantity_rolling_mean' in sales_data.columns else None
                })
            
            return {
                "status": "success",
                "message": f"Detected {len(anomalies)} anomalies in sales data",
                "anomalies": anomalies
            }
            
        except Exception as e:
            logger.error(f"Error detecting sales anomalies: {e}")
            return {
                "status": "error",
                "message": f"Error in anomaly detection: {str(e)}",
                "anomalies": []
            }
    
    # ===== PRICING OPTIMIZATION =====
    
    def suggest_optimal_pricing(self, event_id: str, similar_events_count: int = 5) -> Dict:
        """
        Suggest optimal pricing based on similar events
        
        Args:
            event_id: Event ID to analyze
            similar_events_count: Number of similar events to consider
            
        Returns:
            Dictionary with pricing suggestions
        """
        # Get event data
        event = self._get_event_data(event_id)
        
        if not event:
            return {
                "status": "not_found",
                "message": "Event not found",
                "pricing": {}
            }
        
        # Find similar events
        similar_events = self._find_similar_events(event, similar_events_count)
        
        if not similar_events:
            return {
                "status": "no_similar_events",
                "message": "No similar events found for comparison",
                "pricing": {}
            }
        
        # Extract pricing data
        price_data = []
        for similar_event in similar_events:
            if 'pricing' in similar_event and 'base_price' in similar_event['pricing']:
                price = similar_event['pricing']['base_price']
                
                # Get bookings for this event
                event_bookings = self._get_event_bookings(str(similar_event['_id']))
                ticket_count = sum(booking.get('quantity', 0) for booking in event_bookings)
                
                # Get interactions
                event_interactions = self._get_event_interactions(str(similar_event['_id']))
                interaction_count = len(event_interactions)
                
                # Calculate engagement metrics
                conversion_rate = (ticket_count / interaction_count) * 100 if interaction_count > 0 else 0
                
                price_data.append({
                    "event_id": str(similar_event['_id']),
                    "title": similar_event.get('title', 'Unknown Event'),
                    "price": price,
                    "ticket_count": ticket_count,
                    "interaction_count": interaction_count,
                    "conversion_rate": conversion_rate,
                    "similarity_score": similar_event.get('_similarity_score', 0)
                })
        
        if not price_data:
            return {
                "status": "no_price_data",
                "message": "No pricing data available for similar events",
                "pricing": {}
            }
        
        # Calculate optimal price range
        sorted_by_conversion = sorted(price_data, key=lambda x: x['conversion_rate'], reverse=True)
        top_converting = sorted_by_conversion[:max(1, len(sorted_by_conversion) // 2)]
        
        # Calculate weighted average price
        weighted_sum = sum(item['price'] * item['similarity_score'] for item in top_converting)
        weights_sum = sum(item['similarity_score'] for item in top_converting)
        
        if weights_sum > 0:
            optimal_price = weighted_sum / weights_sum
        else:
            optimal_price = sum(item['price'] for item in top_converting) / len(top_converting)
        
        # Determine price range
        min_price = min(item['price'] for item in top_converting)
        max_price = max(item['price'] for item in top_converting)
        
        # Get current price
        current_price = event.get('pricing', {}).get('base_price', 0)
        
        return {
            "status": "success",
            "message": "Successfully calculated optimal pricing",
            "pricing": {
                "optimal_price": round(optimal_price, 2),
                "price_range": {
                    "min": min_price,
                    "max": max_price
                },
                "current_price": current_price,
                "price_difference": round(optimal_price - current_price, 2) if current_price > 0 else None,
                "similar_events": [
                    {
                        "title": item['title'],
                        "price": item['price'],
                        "conversion_rate": round(item['conversion_rate'], 2)
                    }
                    for item in sorted_by_conversion[:3]  # Top 3 for reference
                ]
            }
        }