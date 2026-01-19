# H&M Two-Tower Recommendation System

A PyTorch implementation of a Two-Tower neural network model for personalized clothing recommendations.

## Overview

This project implements a Two-Tower (dual encoder) recommendation model with separate training and serving components. The model learns separate embeddings for users (customers) and items (articles), enabling fast personalized recommendations via ScaNN similarity search.

### Architecture

- **User Tower**: Encodes customer features into dense embeddings
- **Item Tower**: Encodes article features into dense embeddings
- **Similarity Computation**: Uses dot product similarity between normalized embeddings
- **Fast Search**: ScaNN for recommendation retrieval
- **API**: FastAPI REST service for deployment


---

## Part 1: Training Pipeline

### 1. Setup Training Environment

```bash
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preprocessing

First, preprocess the raw dataset:

```bash
python data_preprocessing.py
```

This will:
- Load the three CSV files from the `data/` folder
- Encode categorical features for customers and articles
- Create positive/negative training samples
- Split data into train/validation sets
- Save preprocessed data to `preprocessed_data/` folder


### 3. Model Training

Train the Two-Tower model:

```bash
python train.py \
    --data_dir ../preprocessed_data \
    --batch_size 512 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --checkpoint_path ../checkpoints/best_model.pt
```

**Key Arguments:**
- `--use_in_batch_negatives`: Use in-batch negative sampling (recommended)
- `--batch_size`: Training batch size (default: 512)
- `--num_epochs`: Number of training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)

---

## Part 2: Production Serving

### 1. Setup Serving Environment

```bash
cd ../serving
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run API Locally

```bash
python api.py
```

### 3. API Endpoints

#### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "customer_id": "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
  "recommendations": [
    {
      "article_id": "0108775015",
      "score": 11.234,
      "rank": 1
    },
    {
      "article_id": "0706016001",
      "score": 10.876,
      "rank": 2
    }
  ],
  "count": 10
}
```

## Dataset

The dataset used for this project was imported from Kaggle and can be found [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). 
