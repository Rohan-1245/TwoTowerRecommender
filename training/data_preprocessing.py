import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path


class DataPreprocessor:
    """
    Preprocesses dataset.
    Handles customer features, article features, and transaction data.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.customer_encoder = LabelEncoder()
        self.article_encoder = LabelEncoder()
        
    def load_data(self):
        self.articles = pd.read_csv(self.data_dir / 'articles.csv')
        self.customers = pd.read_csv(self.data_dir / 'customers.csv')
        self.transactions = pd.read_csv(self.data_dir / 'transactions_train.csv')
        print(f"Loaded {len(self.articles)} articles, {len(self.customers)} customers, {len(self.transactions)} transactions")
        
    def preprocess_articles(self):
        """
        Preprocess article features for the item tower.
        """
        
        # Encode article_id to continuous indices
        self.articles['article_idx'] = self.article_encoder.fit_transform(
            self.articles['article_id']
        )
        
        # Create categorical feature encoders for article attributes
        article_features = {}
        categorical_cols = ['product_type_no', 'product_group_name', 
                          'graphical_appearance_no', 'colour_group_code',
                          'perceived_colour_value_id', 'perceived_colour_master_id',
                          'department_no', 'index_code', 'index_group_no',
                          'section_no', 'garment_group_no']
        
        for col in categorical_cols:
            if col in self.articles.columns:
                # Fill NaN and encode
                self.articles[col] = self.articles[col].fillna('unknown')
                encoder = LabelEncoder()
                article_features[col] = encoder.fit_transform(self.articles[col])
                
        self.article_features = pd.DataFrame(article_features, index=self.articles.index)
        self.article_features['article_idx'] = self.articles['article_idx']
        
        return self.article_features
    
    def preprocess_customers(self):
        """
        Preprocess customer features for the user tower.
        """
        
        # Encode customer_id to continuous indices
        self.customers['customer_idx'] = self.customer_encoder.fit_transform(
            self.customers['customer_id']
        )
        
        # Create categorical feature encoders for customer attributes
        customer_features = {}
        
        # Age (bin into groups)
        if 'age' in self.customers.columns:
            self.customers['age'] = self.customers['age'].fillna(self.customers['age'].median())
            self.customers['age_group'] = pd.cut(self.customers['age'], 
                                                  bins=[0, 20, 30, 40, 50, 60, 100],
                                                  labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60+'])
            encoder = LabelEncoder()
            customer_features['age_group'] = encoder.fit_transform(self.customers['age_group'])
        
        # Other categorical features
        categorical_cols = ['FN', 'Active', 'club_member_status', 'fashion_news_frequency']
        for col in categorical_cols:
            if col in self.customers.columns:
                self.customers[col] = self.customers[col].fillna('unknown')
                encoder = LabelEncoder()
                customer_features[col] = encoder.fit_transform(self.customers[col])
        
        # Postal code (take first 2 digits for region)
        if 'postal_code' in self.customers.columns:
            self.customers['postal_code'] = self.customers['postal_code'].fillna('00000')
            self.customers['postal_region'] = self.customers['postal_code'].astype(str).str[:2]
            encoder = LabelEncoder()
            customer_features['postal_region'] = encoder.fit_transform(self.customers['postal_region'])
        
        self.customer_features = pd.DataFrame(customer_features, index=self.customers.index)
        self.customer_features['customer_idx'] = self.customers['customer_idx']
        
        return self.customer_features
    
    def preprocess_transactions(self, sample_frac=None, min_interactions=5):
        """
        Preprocess transaction data for training.
        
        Args:
            sample_frac: Fraction of data to sample (for faster experimentation)
            min_interactions: Minimum number of interactions per user/item to keep
        """
        
        # Sample if requested
        if sample_frac:
            self.transactions = self.transactions.sample(frac=sample_frac, random_state=42)
            print(f"Sampled to {len(self.transactions)} transactions")
        
        # Map customer_id and article_id to indices
        customer_id_to_idx = dict(zip(self.customers['customer_id'], 
                                      self.customers['customer_idx']))
        article_id_to_idx = dict(zip(self.articles['article_id'], 
                                     self.articles['article_idx']))
        
        self.transactions['customer_idx'] = self.transactions['customer_id'].map(customer_id_to_idx)
        self.transactions['article_idx'] = self.transactions['article_id'].map(article_id_to_idx)
        
        # Remove transactions with unmapped customers or articles
        initial_len = len(self.transactions)
        self.transactions = self.transactions.dropna(subset=['customer_idx', 'article_idx'])
        print(f"Removed {initial_len - len(self.transactions)} transactions with unmapped IDs")
        
        # Filter by minimum interactions
        customer_counts = self.transactions['customer_idx'].value_counts()
        article_counts = self.transactions['article_idx'].value_counts()
        
        valid_customers = customer_counts[customer_counts >= min_interactions].index
        valid_articles = article_counts[article_counts >= min_interactions].index
        
        self.transactions = self.transactions[
            (self.transactions['customer_idx'].isin(valid_customers)) &
            (self.transactions['article_idx'].isin(valid_articles))
        ]
        print(f"After filtering: {len(self.transactions)} transactions, "
              f"{self.transactions['customer_idx'].nunique()} customers, "
              f"{self.transactions['article_idx'].nunique()} articles")
        
        # Convert to int
        self.transactions['customer_idx'] = self.transactions['customer_idx'].astype(int)
        self.transactions['article_idx'] = self.transactions['article_idx'].astype(int)
        
        return self.transactions[['customer_idx', 'article_idx', 't_dat', 'price']]
    
    def create_training_data(self, test_size=0.2, negative_samples_ratio=4):
        """
        Create training and validation datasets with negative sampling.
        
        Args:
            test_size: Fraction of data for validation
            negative_samples_ratio: Number of negative samples per positive sample
        """
        
        # Positive samples (actual transactions)
        positive_samples = self.transactions[['customer_idx', 'article_idx']].copy()
        positive_samples['label'] = 1
        
        # Negative sampling
        num_negatives = len(positive_samples) * negative_samples_ratio
        all_customers = self.transactions['customer_idx'].unique()
        all_articles = self.transactions['article_idx'].unique()
        
        # Create set of positive pairs for efficient lookup
        positive_pairs = set(zip(positive_samples['customer_idx'], 
                               positive_samples['article_idx']))
        
        negative_samples = []
        while len(negative_samples) < num_negatives:
            # Sample random customer-article pairs
            sample_size = min(num_negatives - len(negative_samples), num_negatives // 10)
            customers = np.random.choice(all_customers, size=sample_size)
            articles = np.random.choice(all_articles, size=sample_size)
            
            # Keep only pairs that are not in positive set
            for c, a in zip(customers, articles):
                if (c, a) not in positive_pairs:
                    negative_samples.append({'customer_idx': c, 'article_idx': a, 'label': 0})
                    if len(negative_samples) >= num_negatives:
                        break
        
        negative_samples = pd.DataFrame(negative_samples)
        print(f"Created {len(negative_samples)} negative samples")
        
        # Combine positive and negative samples
        all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
        all_samples = all_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train and validation
        train_data, val_data = train_test_split(all_samples, test_size=test_size, 
                                                random_state=42, stratify=all_samples['label'])
        
        print(f"Training samples: {len(train_data)} (positive: {train_data['label'].sum()})")
        print(f"Validation samples: {len(val_data)} (positive: {val_data['label'].sum()})")
        
        return train_data, val_data
    
    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """Save preprocessed data and encoders."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save feature dataframes
        self.article_features.to_parquet(output_path / 'article_features.parquet')
        self.customer_features.to_parquet(output_path / 'customer_features.parquet')
        
        # Save encoders
        with open(output_path / 'customer_encoder.pkl', 'wb') as f:
            pickle.dump(self.customer_encoder, f)
        with open(output_path / 'article_encoder.pkl', 'wb') as f:
            pickle.dump(self.article_encoder, f)
        
        # Save metadata
        metadata = {
            'num_customers': len(self.customer_encoder.classes_),
            'num_articles': len(self.article_encoder.classes_),
            'customer_feature_dims': {col: self.customer_features[col].max() + 1 
                                     for col in self.customer_features.columns if col != 'customer_idx'},
            'article_feature_dims': {col: self.article_features[col].max() + 1 
                                    for col in self.article_features.columns if col != 'article_idx'}
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved preprocessed data to {output_path}")
        return metadata


if __name__ == '__main__':
    preprocessor = DataPreprocessor(data_dir='data')
    
    preprocessor.load_data()
    
    article_features = preprocessor.preprocess_articles()
    customer_features = preprocessor.preprocess_customers()
    
    transactions = preprocessor.preprocess_transactions(sample_frac=0.1, min_interactions=3)
    
    train_data, val_data = preprocessor.create_training_data(test_size=0.2, negative_samples_ratio=4)
    
    metadata = preprocessor.save_preprocessed_data()
    
    train_data.to_parquet('preprocessed_data/train_data.parquet')
    val_data.to_parquet('preprocessed_data/val_data.parquet')
    
    print(f"Customers: {metadata['num_customers']}")
    print(f"Articles: {metadata['num_articles']}")
