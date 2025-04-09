import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from transformers import BertTokenizer, TFBertForTokenClassification
import json
from f1_score import F1ScoreMetric

# Define the loss function needed for model loading
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    # Create mask for -100 values
    mask = tf.not_equal(y_true, -100)
    
    # Get valid positions
    y_true_valid = tf.boolean_mask(y_true, mask)
    y_pred_valid = tf.boolean_mask(y_pred, mask)
    
    # Compute loss only on valid positions
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)(y_true_valid, y_pred_valid)
    
    return loss

def get_labels():
    """Get label mappings from data.json or use default ones"""
    try:
    
        labels = [
        "O",
        "B-ITEM",
        "I-ITEM",
        "B-QTY",
        "I-QTY",
        "B-PRICE",
        "I-PRICE",
        "B-AMOUNT",
        "I-AMOUNT",
        "B-TOTAL",
        "I-TOTAL",
        "B-TAX",
        "I-TAX",
        "B-TAX_RATE",
        "I-TAX_RATE",
        "B-SUBTOTAL",
        "I-SUBTOTAL"
    ]
        id2label = {i: label for i, label in enumerate(labels)}
        print("Using default label mapping")
        return labels, id2label
    except Exception as e:
        print(f"Could not load labels: {e}")

def load_weights(model, weights_path):
    """Load weights into the model"""
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_weights(weights_path)
        print("Weights loaded successfully")
        return True
    else:
        print(f"Weight file {weights_path} not found")
        return False

def load_model_with_weights(model_path=None, weights_path=None):
    """
    Load a saved model or create a new one and load weights
    
    Args:
        model_path: Path to saved model directory
        weights_path: Path to weights file (.h5)
        
    Returns:
        Loaded model
    """
    # Load labels to determine number of output classes
    labels, id2label = get_labels()
    num_labels = len(labels)
    
    try:
        # Try loading the complete model if path is provided
        if model_path and os.path.isdir(model_path):
            print(f"Loading complete model from {model_path}...")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
                        "TFBertForTokenClassification": TFBertForTokenClassification,
                        "F1ScoreMetric": F1ScoreMetric,
                    }
                )
                print("Complete model loaded successfully")
                return model, id2label
            except Exception as e:
                print(f"Failed to load complete model: {e}")
                print("Will try to recreate model and load weights")
        
        # Recreation approach - create model and load weights
        print("Creating model architecture...")
        
        # Model hyperparameters
        max_length = 128
        dropout_rate = 0.3
        weight_decay = 0.01
        
        # Create model architecture
        from tensorflow.keras import layers
        
        # Load pre-trained BERT
        bert_model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')
        
        # Define inputs
        input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
        
        # Model layers
        bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]

        # Add BiLSTM layers
        bilstm_output = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True)
        )(sequence_output)

        # Add a second BiLSTM layer (optional)
        bilstm_output = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(bilstm_output)

        # Apply dropout again
        bilstm_output = layers.Dropout(0.2)(bilstm_output)

        # Prediction layer for token classification
        logits = layers.Dense(num_labels)(bilstm_output)
        
        # Build model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
        
        # Compile (needed for loading weights)
        model.compile(
            optimizer='adam',
            loss=masked_sparse_categorical_crossentropy,
            metrics=[F1ScoreMetric()]
        )
        
        print("Model architecture created successfully")
        
        # Try to load weights if provided
        if weights_path:
            success = load_weights(model, weights_path)
            if not success and model_path:
                # Try to find weights in the model directory
                alternative_weights = os.path.join(model_path, 'variables', 'variables')
                if os.path.exists(alternative_weights + '.index'):
                    load_weights(model, alternative_weights)
        
        return model, id2label
    
    except Exception as e:
        print(f"Error in model loading process: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def tokenize_text(text, tokenizer, max_length=128):
    """Tokenize input text for prediction"""
    # Split text into words if it's a string
    if isinstance(text, str):
        words = text.split()
    else:
        words = text
    
    # Tokenize and track mapping to original words
    word_tokens = []
    word_token_map = []
    
    for i, word in enumerate(words):
        tokenized_word = tokenizer.tokenize(word)
        word_tokens.extend(tokenized_word)
        word_token_map.extend([i] * len(tokenized_word))
    
    # Add special tokens
    tokens = ['[CLS]'] + word_tokens + ['[SEP]']
    token_map = [-1] + word_token_map + [-1]  # -1 for special tokens
    
    # Convert to IDs and create attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    
    # Pad or truncate
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_map = token_map + ([-1] * padding_length)
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_map = token_map[:max_length]
    
    return {
        "input_ids": np.array([input_ids]),
        "attention_mask": np.array([attention_mask]),
        "token_map": token_map,
        "words": words
    }

def preprocess_text(text):
    """
    Preprocess text by splitting on special characters and ensuring spaces
    """
    import re
    special_char_pattern = r'[()\[\]{}\-;%$£€]'
    
    # First replace special characters with spaced versions
    processed_text = re.sub(f'({special_char_pattern})', r' \1 ', text)
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

def predict_invoice(model, tokenizer, invoice_text, id2label):
    """Make predictions and extract entities from invoice text line by line"""
    # Process input text
    if isinstance(invoice_text, str):
        # Split into lines
        lines = invoice_text.strip().split('\n')
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
    else:
        # Assume it's already a list
        lines = invoice_text
    
    all_words = []
    all_predictions = []
    line_boundaries = []
    current_pos = 0
    
    # Process each line separately
    for line in lines:
        # Preprocess the line to handle special characters
        preprocessed_line = preprocess_text(line)
        words = preprocessed_line.split()
        
        # Tokenize the line
        tokenized = tokenize_text(words, tokenizer)
        
        # Get model predictions for this line
        logits = model.predict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
        predictions = np.argmax(logits, axis=-1)[0]
        
        # Map predictions back to words
        token_map = tokenized["token_map"]
        word_predictions = [None] * len(words)
        
        for j, token_pred in enumerate(predictions):
            word_idx = token_map[j]
            if word_idx != -1 and word_predictions[word_idx] is None:
                word_predictions[word_idx] = id2label[token_pred]
        
        # Store words and predictions for this line
        all_words.extend(words)
        all_predictions.extend(word_predictions)
        
        # Track line boundaries
        line_boundaries.append((current_pos, current_pos + len(words)))
        current_pos += len(words)
    
    # Print token-level predictions by line
    print("\nToken-level predictions (by line):")
    for i, (start, end) in enumerate(line_boundaries):
        print(f"\nLine {i+1}:")
        for j in range(start, end):
            print(f"{all_words[j]}: {all_predictions[j]}")
    
    # Extract entities
    entities = {}
    current_entity = None
    current_value = []
    
    for word, label in zip(all_words, all_predictions):
        if label is None:
            continue
            
        if label.startswith("B-"):
            # If we were building an entity, save it
            if current_entity is not None:
                entity_value = " ".join(current_value)
                if current_entity in entities:
                    if isinstance(entities[current_entity], list):
                        entities[current_entity].append(entity_value)
                    else:
                        entities[current_entity] = [entities[current_entity], entity_value]
                else:
                    entities[current_entity] = entity_value
            
            # Start a new entity
            current_entity = label[2:]  # Remove "B-"
            current_value = [word]
        
        elif label.startswith("I-"):
            entity_type = label[2:]  # Remove "I-"
            if current_entity == entity_type:
                current_value.append(word)
        
        else:  # "O" label
            # If we were building an entity, save it
            if current_entity is not None:
                entity_value = " ".join(current_value)
                if current_entity in entities:
                    if isinstance(entities[current_entity], list):
                        entities[current_entity].append(entity_value)
                    else:
                        entities[current_entity] = [entities[current_entity], entity_value]
                else:
                    entities[current_entity] = entity_value
                
                current_entity = None
                current_value = []
    
    # Save the last entity
    if current_entity is not None:
        entity_value = " ".join(current_value)
        if current_entity in entities:
            if isinstance(entities[current_entity], list):
                entities[current_entity].append(entity_value)
            else:
                entities[current_entity] = [entities[current_entity], entity_value]
        else:
            entities[current_entity] = entity_value
    
    return entities

def format_as_table(entities):
    """Format extracted entities as a table"""
    # Create dictionary for DataFrame
    df_data = {}
    
    # Process different entity types
    for entity_type in ["ITEM", "QTY", "PRICE", "AMOUNT", "SUBTOTAL", "TAX", "TAX_RATE", "TOTAL"]:
        if entity_type in entities:
            values = entities[entity_type] if isinstance(entities[entity_type], list) else [entities[entity_type]]
            df_data[entity_type] = values
    
    # Create DataFrame if we have data
    if df_data:
        # Get max length and pad shorter lists
        max_len = max([len(v) for v in df_data.values()])
        for k in df_data:
            if len(df_data[k]) < max_len:
                df_data[k] = df_data[k] + [""] * (max_len - len(df_data[k]))
        
        # Create and return DataFrame
        return pd.DataFrame(df_data)
    
    return "No structured data could be extracted"

def main():
    parser = argparse.ArgumentParser(description="Test a trained invoice NER model on sample data")
    parser.add_argument("--model", type=str, default="/invoice_bert_sequence_labeling",
                       help="Path to saved model directory")
    parser.add_argument("--weights", type=str, default="/best_model_weights.h5",
                       help="Path to model weights file (.h5)")
    parser.add_argument("--tokenizer", type=str, default="/invoice_bert_tokenizer",
                       help="Path to tokenizer directory or 'bert-base-uncased' to use default")
    parser.add_argument("--invoice", type=str, 
                       default=None,
                       help="Invoice text to process (uses sample if not provided)")
    
    args = parser.parse_args()
    
    # Replace the hardcoded sample_invoice with multi-line user input
    print("Please enter the invoice text to process (or press Enter to use a default sample)")
    print("Enter your text (press Ctrl+D on Mac/Linux or Ctrl+Z on Windows + Enter when done):")
    
    # Collect lines until EOF (Ctrl+D/Ctrl+Z) or empty line
    lines = []
    try:
        while True:
            line = input()
            if not line:  # Empty line indicates end of input
                break
            lines.append(line)
    except EOFError:  # Handles Ctrl+D/Ctrl+Z
        pass

    user_input_invoice = '\n'.join(lines).strip()

    # Default sample invoice if the user doesn't provide input
    sample_invoice = user_input_invoice if user_input_invoice else """
    SAMPLE INVOICE
    Item 1    2    $10.00    $20.00
    Item 2    1    $15.00    $15.00
    Subtotal: $35.00
    Tax (10%): $3.50
    Total: $38.50
    """

    invoice_text = args.invoice if args.invoice else sample_invoice
    
    # Load tokenizer
    try:
        if args.tokenizer == "bert-base-uncased":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using default BERT tokenizer instead")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load model and weights
    model, id2label = load_model_with_weights(args.model, args.weights)
    
    if model is None:
        print("Failed to load or create model. Exiting.")
        return
    
    # Display invoice text
    print("\n" + "="*50)
    print("Invoice Text:")
    print(invoice_text)
    print("="*50)
    
    # Make predictions
    entities = predict_invoice(model, tokenizer, invoice_text, id2label)
    
    # Display extracted entities
    print("\nExtracted Entities:")
    for entity, value in entities.items():
        print(f"{entity}: {value}")
    
    # Display as table
    print("\nFormatted as Table:")
    table = format_as_table(entities)
    print(table)

if __name__ == "__main__":
    main()
