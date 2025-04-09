# BERT Invoice Processing Model

## Overview
This project implements a Named Entity Recognition (NER) model for processing invoice data using BERT (Bidirectional Encoder Representations from Transformers) with additional BiLSTM layers. The model identifies and extracts key information from invoices such as items, quantities, prices, amounts, taxes, and totals.

## Model Architecture
- Base: BERT (bert-base-uncased)
- Additional layers:
  - Bidirectional LSTM (256 units)
  - Bidirectional LSTM (128 units)
  - Dropout (0.2)
  - Dense layer for token classification

## Features
- Named Entity Recognition for invoice data
- Multi-label classification
- Custom F1-score metric implementation
- Support for both training and inference
- Handles multi-line invoice text
- Preserves special characters in tokenization

## Labels
The model recognizes the following entities:
```python
- O (Outside)
- B-ITEM/I-ITEM (Beginning/Inside of Item)
- B-QTY/I-QTY (Quantity)
- B-PRICE/I-PRICE (Price)
- B-AMOUNT/I-AMOUNT (Amount)
- B-TOTAL/I-TOTAL (Total)
- B-TAX/I-TAX (Tax)
- B-TAX_RATE/I-TAX_RATE (Tax Rate)
- B-SUBTOTAL/I-SUBTOTAL (Subtotal)
```

## Requirements
Model ran on MACOS and tested with INTEL Core processor
```
    "matplotlib>=3.10.0",
    "numpy>=1.24.3",
    "pandas>=2.1.0",
    "opencv-python>=4.8.1.78",
    "pyyaml==5.3.1",
    "pytesseract>=0.3.13",
    "scikit-learn>=1.6.1",
    "tensorflow==2.13.1",
    "transformers>=4.49.0",
```

## Usage

### Setting up
    project is set with uv
    to set up the project you can run in the terminal:
``` 
    uv init
    uv sync
```

### Training
```
    uv run main.py
```

### Testing
```
    uv run test_model.py
```


### Processing an Invoice
```python
from test_model import load_model_with_weights, predict_invoice
from transformers import BertTokenizer

# Load model and tokenizer
model, id2label = load_model_with_weights("path/to/model", "path/to/weights")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Process invoice
invoice_text = """
SAMPLE INVOICE
Item 1    2    $10.00    $20.00
Item 2    1    $15.00    $15.00
Subtotal: $35.00
"""
entities = predict_invoice(model, tokenizer, invoice_text, id2label)
```

## Model Performance
- Early stopping with patience of 5 epochs
- Learning rate scheduling with ReduceLROnPlateau
- Evaluation metrics:
  - F1-score (micro, macro, weighted)
  - Classification report for each entity type

## File Structure
```
BERT_model/
├── main.py           # Training script
├── test_model.py     # Testing and inference
├── f1_score.py       # Custom F1 metric
└── README.md         # Documentation
```

## Training Parameters
- Batch size: 21
- Maximum sequence length: 128
- Learning rate: 3e-5
- Dropout rate: 0.5
- Weight decay: 0.02

## Contributing
Feel free to submit issues and enhancement requests.

