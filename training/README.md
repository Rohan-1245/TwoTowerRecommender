# Recommendation Model - Training

Training folder for the Two-Tower recommendation model, including data proprocessing, model configuration, and training.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python data_preprocessing.py

# Train model
python train.py --batch_size 512 --num_epochs 20
```

**Key Arguments:**
- `--use_in_batch_negatives`: Use in-batch negative sampling (recommended)
- `--batch_size`: Batch size
- `--num_epochs`: Number of epochs
- `--learning_rate`: Learning rate
- `--patience`: Early stopping patience

