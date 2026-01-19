from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import pandas as pd
from pathlib import Path
from inference_engine import RecommendationInference, load_model_for_inference
import uvicorn


class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID to generate recommendations for")
    top_k: int = Field(10, description="Number of recommendations to return", ge=1, le=100)
    exclude_items: Optional[List[str]] = Field(None, description="List of item IDs to exclude from recommendations")


class RecommendationItem(BaseModel):
    article_id: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[RecommendationItem]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_items: int
    num_customers: int


app = FastAPI(
    title="Recommendation API",
    description="Two-Tower recommendation system for personalized clothing recommendations",
    version="1.0.0"
)

inference_engine = None
customer_id_to_idx = None
article_idx_to_id = None


@app.on_event("startup")
async def load_model():
    """Load model and data on startup."""
    global inference_engine, customer_id_to_idx, article_idx_to_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path('../checkpoints/best_model.pt')
    metadata_path = Path('../preprocessed_data/metadata.pkl')
    data_dir = Path('../preprocessed_data')
    
    model, metadata = load_model_for_inference(
        str(checkpoint_path),
        str(metadata_path),
        device=device
    )
    
    customer_features = pd.read_parquet(data_dir / 'customer_features.parquet')
    article_features = pd.read_parquet(data_dir / 'article_features.parquet')
    
    customers = pd.read_csv(data_dir.parent / 'data' / 'customers.csv')
    articles = pd.read_csv(data_dir.parent / 'data' / 'articles.csv')
    
    customer_id_to_idx = dict(zip(customers['customer_id'], customer_features['customer_idx']))
    article_idx_to_id = dict(zip(article_features['article_idx'], articles['article_id']))
    
    inference_engine = RecommendationInference(
        model, customer_features, article_features, metadata, device=device
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "H&M Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        num_items=len(inference_engine.article_indices),
        num_customers=len(inference_engine.customer_features_df)
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Generate personalized recommendations for a customer.
    
    Args:
        request: RecommendationRequest with customer_id and optional parameters
    
    Returns:
        RecommendationResponse with ranked recommendations
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.customer_id not in customer_id_to_idx:
        raise HTTPException(
            status_code=404, 
            detail=f"Customer ID '{request.customer_id}' not found"
        )
    
    customer_idx = customer_id_to_idx[request.customer_id]
    
    # Convert exclude_items from IDs to indices if provided
    exclude_indices = None
    if request.exclude_items:
        article_id_to_idx = {v: k for k, v in article_idx_to_id.items()}
        exclude_indices = set()
        for item_id in request.exclude_items:
            if item_id in article_id_to_idx:
                exclude_indices.add(article_id_to_idx[item_id])
    
    try:
        recommendations = inference_engine.recommend_for_user(
            customer_idx, 
            top_k=request.top_k,
            exclude_items=exclude_indices
        )
        
        recommendation_items = []
        for rank, (article_idx, score) in enumerate(recommendations, 1):
            article_id = article_idx_to_id.get(article_idx, str(article_idx))
            recommendation_items.append(
                RecommendationItem(
                    article_id=article_id,
                    score=float(score),
                    rank=rank
                )
            )
        
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recommendation_items,
            count=len(recommendation_items)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
