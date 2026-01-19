import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from model import create_model


class RecommendationDataset(Dataset):
    """
    Concatenate customer and article features for singular training Dataset.
    """
    
    def __init__(self, interactions_df, customer_features_df, article_features_df):
        """
        Args:
            interactions_df: DataFrame with columns [customer_idx, article_idx, label]
            customer_features_df: DataFrame with customer features
            article_features_df: DataFrame with article features
        """
        self.interactions = interactions_df.reset_index(drop=True)
        self.customer_features_df = customer_features_df
        self.article_features_df = article_features_df
        
        # Get feature column names (excluding idx columns)
        self.customer_feature_cols = [col for col in customer_features_df.columns 
                                      if col != 'customer_idx']
        self.article_feature_cols = [col for col in article_features_df.columns 
                                    if col != 'article_idx']
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        customer_idx = int(row['customer_idx'])
        article_idx = int(row['article_idx'])
        label = int(row['label'])
        
        # Get customer features
        customer_row = self.customer_features_df[
            self.customer_features_df['customer_idx'] == customer_idx
        ].iloc[0]
        customer_features = {
            col: int(customer_row[col]) 
            for col in self.customer_feature_cols
        }
        
        # Get article features
        article_row = self.article_features_df[
            self.article_features_df['article_idx'] == article_idx
        ].iloc[0]
        article_features = {
            col: int(article_row[col]) 
            for col in self.article_feature_cols
        }
        
        return customer_features, article_features, label


def collate_fn(batch):
    """
    Custom collate function to handle dict features.
    """
    customer_features_list, article_features_list, labels = zip(*batch)
    
    # Stack customer features
    customer_features = {}
    for key in customer_features_list[0].keys():
        customer_features[key] = torch.tensor([f[key] for f in customer_features_list])
    
    # Stack article features
    article_features = {}
    for key in article_features_list[0].keys():
        article_features[key] = torch.tensor([f[key] for f in article_features_list])
    
    labels = torch.tensor(labels)
    
    return customer_features, article_features, labels


class Trainer:
    """
    Trainer for Two-Tower recommendation model.
    """
    
    def __init__(self, model, loss_fn, optimizer, device='cuda', 
                 use_in_batch_negatives=False):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.use_in_batch_negatives = use_in_batch_negatives
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for customer_features, article_features, labels in pbar:
            # Move to device
            customer_features = {k: v.to(self.device) for k, v in customer_features.items()}
            article_features = {k: v.to(self.device) for k, v in article_features.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_in_batch_negatives:
                # Only use positive samples for in-batch negatives
                pos_mask = labels == 1
                if pos_mask.sum() == 0:
                    continue
                    
                customer_features_pos = {k: v[pos_mask] for k, v in customer_features.items()}
                article_features_pos = {k: v[pos_mask] for k, v in article_features.items()}
                
                similarity_matrix = self.model(customer_features_pos, article_features_pos)
                loss = self.loss_fn(similarity_matrix)
            else:
                scores = self.model(customer_features, article_features)
                loss = self.loss_fn(scores, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Evaluating')
            for customer_features, article_features, labels in pbar:
                # Move to device
                customer_features = {k: v.to(self.device) for k, v in customer_features.items()}
                article_features = {k: v.to(self.device) for k, v in article_features.items()}
                labels = labels.to(self.device)
                
                if self.use_in_batch_negatives:
                    pos_mask = labels == 1
                    if pos_mask.sum() == 0:
                        continue
                    customer_features_pos = {k: v[pos_mask] for k, v in customer_features.items()}
                    article_features_pos = {k: v[pos_mask] for k, v in article_features.items()}
                    similarity_matrix = self.model(customer_features_pos, article_features_pos)
                    loss = self.loss_fn(similarity_matrix)
                else:
                    scores = self.model(customer_features, article_features)
                    loss = self.loss_fn(scores, labels)
                    
                    # Collect predictions for metrics
                    preds = torch.sigmoid(scores)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        if not self.use_in_batch_negatives and len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # AUC and accuracy
            from sklearn.metrics import roc_auc_score, accuracy_score
            auc = roc_auc_score(all_labels, all_preds)
            accuracy = accuracy_score(all_labels, all_preds > 0.5)
            
            return avg_loss, auc, accuracy
        
        return avg_loss, None, None
    
    def save_checkpoint(self, filepath, epoch, best_loss):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = Path(args.data_dir)
    
    train_data = pd.read_parquet(data_dir / 'train_data.parquet')
    val_data = pd.read_parquet(data_dir / 'val_data.parquet')
    customer_features = pd.read_parquet(data_dir / 'customer_features.parquet')
    article_features = pd.read_parquet(data_dir / 'article_features.parquet')
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    train_dataset = RecommendationDataset(train_data, customer_features, article_features)
    val_dataset = RecommendationDataset(val_data, customer_features, article_features)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    model, loss_fn = create_model(
        metadata,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        temperature=args.temperature,
        use_in_batch_negatives=args.use_in_batch_negatives
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                weight_decay=args.weight_decay)
    
    trainer = Trainer(model, loss_fn, optimizer, device=device,
                     use_in_batch_negatives=args.use_in_batch_negatives)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_results = trainer.evaluate(val_loader)
        if len(val_results) == 3:
            val_loss, val_auc, val_acc = val_results
            print(f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
        else:
            val_loss = val_results
            print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(args.checkpoint_path, epoch, best_val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Two-Tower Recommendation Model')
    
    parser.add_argument('--data_dir', type=str, default='preprocessed_data',
                       help='Directory with preprocessed data')
    
    parser.add_argument('--embedding_dim', type=int, default=32,
                       help='Dimension for feature embeddings')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Final embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.05,
                       help='Temperature for scaling similarities')
    parser.add_argument('--use_in_batch_negatives', action='store_true',
                       help='Use in-batch negative sampling')
    
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pt',
                       help='Path to save best model')
    
    args = parser.parse_args()

    Path(args.checkpoint_path).parent.mkdir(exist_ok=True, parents=True)
    
    main(args)
