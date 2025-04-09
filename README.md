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
Model ran on MACOS and tested with INTEL Core processor.
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
Project is set with uv. To set up the project you can run in the terminal:
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
Office Supplies 10 $5.99 $59.90
Printer Paper 5 $12.50 $62.50
Ink Cartridges 3 $35.00 $105.00
Total Amount: $227.40
Tax Rate: 8%
Tax: $16.80
"""
entities = predict_invoice(model, tokenizer, invoice_text, id2label)
```

### Example results
```
    Token-level predictions (by line):

    Line 1:
    Office: B-ITEM
    Supplies: I-ITEM
    10: B-QTY
    $: O
    5.99: B-PRICE
    $: O
    59.90: B-AMOUNT

    Line 2:
    Printer: B-ITEM
    Paper: I-ITEM
    5: B-QTY
    $: O
    12.50: B-PRICE
    $: O
    62.50: B-AMOUNT

    Line 3:
    Ink: B-ITEM
    Cartridges: I-ITEM
    3: B-QTY
    $: O
    35.00: B-PRICE
    $: O
    105.00: B-AMOUNT

    Line 4:
    Total: O
    Amount:: O
    $: B-TOTAL
    227.40: B-TOTAL

    Line 5:
    Tax: O
    Rate:: O
    8: O
    %: O

    Line 6:
    Tax:: O
    $: O
    16.80: O

    Extracted Entities:
    ITEM: ['Office Supplies', 'Printer Paper', 'Ink Cartridges']
    QTY: ['10', '5', '3']
    PRICE: ['5.99', '12.50', '35.00']
    AMOUNT: ['59.90', '62.50', '105.00']
    TOTAL: ['$', '227.40']

    Formatted as Table:
                ITEM QTY  PRICE  AMOUNT   TOTAL
    0  Office Supplies  10   5.99   59.90       $
    1    Printer Paper   5  12.50   62.50  227.40
    2   Ink Cartridges   3  35.00  105.00
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

