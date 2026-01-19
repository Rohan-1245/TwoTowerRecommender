import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerEmbedding(nn.Module):
    """
    Generic tower for embedding categorical features.
    Supports multiple categorical features with different cardinalities.
    """
    
    def __init__(self, feature_dims, embedding_dim=32, hidden_dims=[128, 64]):
        """
        Args:
            feature_dims: Dict mapping feature names to their cardinalities
            embedding_dim: Dimension for each feature embedding
            hidden_dims: List of hidden layer dimensions
        """
        super(TowerEmbedding, self).__init__()
        
        self.feature_names = list(feature_dims.keys())
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in feature_dims.items()
        })
        
        input_dim = len(feature_dims) * embedding_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, features):
        """
        Args:
            features: Dict mapping feature names to tensor of indices
        Returns:
            Embedded representation
        """

        embeddings = []
        for name in self.feature_names:
            emb = self.embeddings[name](features[name])
            embeddings.append(emb)
        
        x = torch.cat(embeddings, dim=1)
        
        x = self.mlp(x)
        
        return x


class UserTower(nn.Module):
    """
    User tower for encoding customer features.
    """
    
    def __init__(self, customer_feature_dims, embedding_dim=32, 
                 hidden_dims=[128, 64], output_dim=64):
        """
        Args:
            customer_feature_dims: Dict of customer feature cardinalities
            embedding_dim: Dimension for each feature embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Final output dimension for the tower
        """
        super(UserTower, self).__init__()
        
        self.tower = TowerEmbedding(customer_feature_dims, embedding_dim, hidden_dims)
        self.projection = nn.Linear(self.tower.output_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, customer_features):
        """
        Args:
            customer_features: Dict of customer feature tensors
        Returns:
            User embedding of shape (batch_size, output_dim)
        """
        x = self.tower(customer_features)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ItemTower(nn.Module):
    """
    Item tower for encoding article features.
    """
    
    def __init__(self, article_feature_dims, embedding_dim=32, 
                 hidden_dims=[128, 64], output_dim=64):
        """
        Args:
            article_feature_dims: Dict of article feature cardinalities
            embedding_dim: Dimension for each feature embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Final output dimension for the tower
        """
        super(ItemTower, self).__init__()
        
        self.tower = TowerEmbedding(article_feature_dims, embedding_dim, hidden_dims)
        self.projection = nn.Linear(self.tower.output_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, article_features):
        """
        Args:
            article_features: Dict of article feature tensors
        Returns:
            Item embedding of shape (batch_size, output_dim)
        """
        x = self.tower(article_features)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class TwoTowerModel(nn.Module):
    """
    Two-Tower recommendation model.
    Separately encodes users and items, then computes similarity scores.
    """
    
    def __init__(self, customer_feature_dims, article_feature_dims,
                 embedding_dim=32, hidden_dims=[128, 64], output_dim=64,
                 temperature=0.05):
        """
        Args:
            customer_feature_dims: Dict of customer feature cardinalities
            article_feature_dims: Dict of article feature cardinalities
            embedding_dim: Dimension for feature embeddings
            hidden_dims: Hidden layer dimensions for towers
            output_dim: Final embedding dimension
            temperature: Temperature for scaling dot products
        """
        super(TwoTowerModel, self).__init__()
        
        self.user_tower = UserTower(
            customer_feature_dims, 
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        self.item_tower = ItemTower(
            article_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        self.temperature = temperature
        
    def forward(self, customer_features, article_features):
        """
        Args:
            customer_features: Dict of customer feature tensors
            article_features: Dict of article feature tensors
        Returns:
            Similarity scores (logits)
        """
        user_embeddings = self.user_tower(customer_features)
        item_embeddings = self.item_tower(article_features)
        
        scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        scores = scores / self.temperature
        
        return scores
    
    def get_user_embeddings(self, customer_features):
        """Get user embeddings for inference."""
        return self.user_tower(customer_features)
    
    def get_item_embeddings(self, article_features):
        """Get item embeddings for inference."""
        return self.item_tower(article_features)
    
    def predict_batch(self, user_embeddings, item_embeddings):
        """
        Compute scores for all user-item pairs in batch.
        
        Args:
            user_embeddings: (num_users, output_dim)
            item_embeddings: (num_items, output_dim)
        Returns:
            scores: (num_users, num_items)
        """
        scores = torch.matmul(user_embeddings, item_embeddings.T)
        scores = scores / self.temperature
        return scores


class TwoTowerLoss(nn.Module):
    """
    Loss function for Two-Tower model.
    Uses binary cross-entropy for positive/negative samples.
    """
    
    def __init__(self):
        super(TwoTowerLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, scores, labels):
        """
        Args:
            scores: Predicted scores (logits)
            labels: Ground truth labels (0 or 1)
        Returns:
            Loss value
        """
        return self.bce_loss(scores, labels.float())


class TwoTowerWithInBatchNegatives(nn.Module):
    """
    Two-Tower model with in-batch negative sampling.
    More efficient training by treating other items in batch as negatives.
    """
    
    def __init__(self, customer_feature_dims, article_feature_dims,
                 embedding_dim=32, hidden_dims=[128, 64], output_dim=64,
                 temperature=0.05):
        super(TwoTowerWithInBatchNegatives, self).__init__()
        
        self.user_tower = UserTower(
            customer_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        self.item_tower = ItemTower(
            article_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        self.temperature = temperature
        
    def forward(self, customer_features, article_features):
        """
        Compute similarity matrix for all user-item pairs in batch.
        
        Args:
            customer_features: Dict of customer feature tensors (batch_size,)
            article_features: Dict of article feature tensors (batch_size,)
        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        user_embeddings = self.user_tower(customer_features)
        item_embeddings = self.item_tower(article_features)

        similarity_matrix = torch.matmul(user_embeddings, item_embeddings.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def get_user_embeddings(self, customer_features):
        """Get user embeddings for inference."""
        return self.user_tower(customer_features)
    
    def get_item_embeddings(self, article_features):
        """Get item embeddings for inference."""
        return self.item_tower(article_features)


class InBatchNegativeLoss(nn.Module):
    """
    Loss for in-batch negative sampling.
    Treats diagonal as positive pairs, off-diagonal as negatives.
    """
    
    def __init__(self):
        super(InBatchNegativeLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, similarity_matrix):
        """
        Args:
            similarity_matrix: (batch_size, batch_size) similarity scores
        Returns:
            Loss value
        """
        batch_size = similarity_matrix.shape[0]
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        loss = self.cross_entropy(similarity_matrix, labels)
        
        return loss


def create_model(metadata, embedding_dim=32, hidden_dims=[128, 64], 
                output_dim=64, temperature=0.05, use_in_batch_negatives=False):
    """
    Factory function to create a Two-Tower model from metadata.
    
    Args:
        metadata: Dict containing feature dimensions
        embedding_dim: Dimension for feature embeddings
        hidden_dims: Hidden layer dimensions
        output_dim: Final embedding dimension
        temperature: Temperature for scaling
        use_in_batch_negatives: Whether to use in-batch negative sampling
    Returns:
        model, loss_fn
    """
    customer_feature_dims = metadata['customer_feature_dims']
    article_feature_dims = metadata['article_feature_dims']
    
    if use_in_batch_negatives:
        model = TwoTowerWithInBatchNegatives(
            customer_feature_dims,
            article_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            temperature=temperature
        )
        loss_fn = InBatchNegativeLoss()
    else:
        model = TwoTowerModel(
            customer_feature_dims,
            article_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            temperature=temperature
        )
        loss_fn = TwoTowerLoss()
    
    return model, loss_fn
