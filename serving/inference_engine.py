import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from model import create_model
import scann


class RecommendationInference:
    """
    Inference engine for Two-Tower recommendation model.
    Uses ScaNN for fast similarity search.
    """
    
    def __init__(self, model, customer_features_df, article_features_df, 
                 metadata, device='cuda'):
        """
        Args:
            model: Trained Two-Tower model
            customer_features_df: DataFrame with customer features
            article_features_df: DataFrame with article features
            metadata: Metadata dict with feature dimensions
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.customer_features_df = customer_features_df
        self.article_features_df = article_features_df
        self.metadata = metadata
        
        self.customer_feature_cols = [col for col in customer_features_df.columns 
                                      if col != 'customer_idx']
        self.article_feature_cols = [col for col in article_features_df.columns 
                                    if col != 'article_idx']
        
        print("Pre-computing item embeddings...")
        self.item_embeddings = self._compute_all_item_embeddings()
        self.article_indices = self.article_features_df['article_idx'].values
        
        print("Building ScaNN index...")
        self.searcher = self._build_scann_index()
        
    def _compute_all_item_embeddings(self, batch_size=1024):
        """Pre-compute embeddings for all items."""
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.article_features_df), batch_size)):
                batch_df = self.article_features_df.iloc[i:i+batch_size]
                
                article_features = {
                    col: torch.tensor(batch_df[col].values).to(self.device)
                    for col in self.article_feature_cols
                }
                
                embeddings = self.model.get_item_embeddings(article_features)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def _build_scann_index(self):
        """Build ScaNN index for fast similarity search."""
        embeddings_np = self.item_embeddings.numpy().astype(np.float32)
        
        # Build ScaNN searcher
        searcher = (
            scann.scann_ops_pybind.builder(embeddings_np, 10, "dot_product")
            .tree(
                num_leaves=2000,
                num_leaves_to_search=100,
                training_sample_size=min(250000, len(embeddings_np))
            )
            .score_ah(
                2,
                anisotropic_quantization_threshold=0.2
            )
            .reorder(100)
            .build()
        )
        
        return searcher
    
    def get_user_embedding(self, customer_idx):
        """Get embedding for a single user."""
        customer_row = self.customer_features_df[
            self.customer_features_df['customer_idx'] == customer_idx
        ].iloc[0]
        
        customer_features = {
            col: torch.tensor([int(customer_row[col])]).to(self.device)
            for col in self.customer_feature_cols
        }
        
        with torch.no_grad():
            user_embedding = self.model.get_user_embeddings(customer_features)
        
        return user_embedding
    
    def recommend_for_user(self, customer_idx, top_k=10, exclude_items=None):
        """
        Generate top-k recommendations for a user using ScaNN.
        
        Args:
            customer_idx: Customer index
            top_k: Number of recommendations
            exclude_items: Set of article indices to exclude
        Returns:
            List of (article_idx, score) tuples
        """
        user_embedding = self.get_user_embedding(customer_idx)
        user_emb_np = user_embedding.cpu().numpy().astype(np.float32).reshape(-1)
        
        # ScaNN search
        if exclude_items:
            retrieve_k = min(top_k + len(exclude_items) * 2, len(self.article_indices))
            indices, distances = self.searcher.search(user_emb_np, final_num_neighbors=retrieve_k)
            
            # Filter excluded items
            mask = ~np.isin(self.article_indices[indices], list(exclude_items))
            indices = indices[mask][:top_k]
            distances = distances[mask][:top_k]
        else:
            indices, distances = self.searcher.search(user_emb_np, final_num_neighbors=top_k)
        
        top_articles = self.article_indices[indices]
        top_scores = distances
        
        return list(zip(top_articles, top_scores))


def load_model_for_inference(checkpoint_path, metadata_path, device='cuda'):
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        metadata_path: Path to metadata pickle file
        device: Device to load model on
    Returns:
        Loaded model
    """
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    model, _ = create_model(metadata)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, metadata
