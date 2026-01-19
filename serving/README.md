# Recommendation API - Serving

Serving folder for the Two-Tower recommendation model, including the inference engine as well as the API deployment.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py
```

The API will start on `http://localhost:8000`

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t recsys-api .
docker run -p 8000:8000 \
  -v $(pwd)/../checkpoints:/app/checkpoints:ro \
  -v $(pwd)/../preprocessed_data:/app/preprocessed_data:ro \
  -v $(pwd)/../data:/app/data:ro \
  recsys-api
```

## API Endpoints

### POST /recommend
Generate recommendations for a customer

**Request:**
```json
{
  "customer_id": "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
  "top_k": 10,
  "exclude_items": ["0108775015"]  // optional
}
```

**Response:**
```json
{
  "customer_id": "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
  "recommendations": [
    {
      "article_id": "0706016001",
      "score": 11.234,
      "rank": 1
    }
  ],
  "count": 10
}
```
