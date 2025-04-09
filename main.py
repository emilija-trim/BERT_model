import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from transformers import TFBertForTokenClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import re

# 1. Define the labels for invoice data
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

# Create label to ID mapping
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)

# 2. Load and preprocess data
# Replace the load_sample_data function with this one
def load_sample_data():
    """
    Load data from the converted JSON file.
    """
    import json
    
    try:
        # Try to load the converted data
        with open('/training_data/', 'r') as f:
            data = json.load(f)
        
        # Extract texts and tags from the data
        processed_data = []
        
        if isinstance(data, dict) and 'data' in data:
            # If data is a dict with 'data' key
            for item in data['data']:
                processed_data.append((" ".join(item['tokens']), item['tags']))

            # Update global labels if needed
            global labels, label2id, id2label, num_labels
            if 'labels' in data:
                labels = data['labels']
                label2id = data['label2id'] if 'label2id' in data else {label: i for i, label in enumerate(labels)}
                id2label = {int(k): v for k, v in data['id2label'].items()} if 'id2label' in data else {i: label for i, label in enumerate(labels)}
                num_labels = len(labels)
                print(f"Updated label mappings with {num_labels} labels")
        
        print(f"Loaded {len(processed_data)} examples from converted data")
        return processed_data
    
    except Exception as e:
        print(f"Error loading converted data: {e}")
        print("Falling back to sample data")
        
        # Fall back to the original sample data
        sample_data = [
            ("Office Supplies 10 $5.99 $59.90", ["B-ITEM", "I-ITEM", "B-QTY", "B-PRICE", "B-AMOUNT"]),
            ("Printer Paper 5 $12.50 $62.50", ["B-ITEM", "I-ITEM", "B-QTY", "B-PRICE", "B-AMOUNT"]),
        ]
        return sample_data

# Load the data
data = load_sample_data()
print(f"============Data============\n{data[:3]}")
  
texts = [item[0].split() for item in data]
tags = [item[1] for item in data]

# 3. Tokenize the data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Adjust based on your data

def tokenize_and_align_labels(texts, tags):
    tokenized_inputs = []
    aligned_labels = []
    
    for i, (sentence, word_labels) in enumerate(zip(texts, tags)):
        # Tokenize each word and count resulting subword tokens
        word_tokens = []
        word_label_ids = []
        
        for word, label in zip(sentence, word_labels):
            # Tokenize the word and count # of subwords
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            
            # Add the tokenized word to the final tokenized word list
            word_tokens.extend(tokenized_word)
            
            # Add the same label to each subword of the word
            label_id = label2id[label]
            word_label_ids.extend([label_id] + [label_id] * (n_subwords - 1))
        
        # Add [CLS] and [SEP] tokens
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + word_tokens + ['[SEP]'])
        attention_mask = [1] * len(input_ids)
        
        # Pad or truncate to max_length
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        
        # Create label ids with -100 for special tokens (CLS, SEP, PAD)
        # BERT standard practice: -100 means "ignore this token in loss calculation"
        label_ids = [-100] + word_label_ids + [-100]  # CLS + tokens + SEP
        
        # Pad or truncate labels
        if padding_length > 0:
            label_ids = label_ids + ([-100] * padding_length)
        else:
            label_ids = label_ids[:max_length]
        
        tokenized_inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        aligned_labels.append(label_ids)

    # print("============ Tokenized Inputs ============")
    # print(tokenized_inputs)
    # print("============ Aligned Labels ============")
    # print(aligned_labels)
    
    return tokenized_inputs, aligned_labels

# Process data
tokenized_inputs, aligned_labels = tokenize_and_align_labels(texts, tags)

# Convert to TensorFlow dataset
def create_tf_dataset(tokenized_inputs, aligned_labels, batch_size=8):
    input_ids = np.array([item["input_ids"] for item in tokenized_inputs])
    attention_mask = np.array([item["attention_mask"] for item in tokenized_inputs])
    labels = np.array(aligned_labels)
    
    dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, labels))
    
    dataset = dataset.shuffle(len(input_ids)).batch(batch_size)
    return dataset

# Split data into train and validation
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    tokenized_inputs, aligned_labels, test_size=0.2, random_state=42
)

batch_size = 8
train_dataset = create_tf_dataset(train_inputs, train_labels, batch_size)
val_dataset = create_tf_dataset(val_inputs, val_labels, batch_size)

# 4. Build the BERT sequence labeling model
# Define a custom F1-score metric for model training
class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        # True positives, false positives, false negatives
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Create mask for -100 values
        mask = tf.not_equal(y_true, -100)
        # Get valid positions
        y_true_valid = tf.boolean_mask(y_true, mask)
        
        # Get predicted classes from logits
        y_pred_valid = tf.boolean_mask(tf.argmax(y_pred, axis=-1), mask)
        
        # One-hot encode for multi-class F1 calculation
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true_valid, tf.int32), num_classes)
        y_pred_oh = tf.one_hot(y_pred_valid, num_classes)
        
        # Update tp, fp, fn
        tp_update = tf.reduce_sum(y_true_oh * y_pred_oh)
        fp_update = tf.reduce_sum(y_pred_oh) - tp_update
        fn_update = tf.reduce_sum(y_true_oh) - tp_update
        
        self.tp.assign_add(tp_update)
        self.fp.assign_add(fp_update)
        self.fn.assign_add(fn_update)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return f1
    
    def reset_state(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)

# Update the build_model function to remove accuracy metric
def build_model(dropout_rate=0.3, weight_decay=0.01):
    """
    Build the BERT sequence labeling model with regularization
    
    Args:
        dropout_rate: Dropout rate to prevent overfitting (default: 0.3)
        weight_decay: L2 regularization coefficient (default: 0.01)
    
    Returns:
        Compiled model
    """
    # Load pre-trained BERT model
    bert_model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')
    
    # Define inputs
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    
    # Pass inputs through BERT
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
    sequence_output = bert_outputs[0]  # The last hidden state

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
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits
    )
    
    # Compile with the same loss and apply weight decay to optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=3e-5,
        weight_decay=weight_decay # Apply weight decay to optimizer (for TF 2.11+)
    )
    
    # For older TensorFlow versions where Adam doesn't support weight_decay parameter
    try:
        model.compile(
            optimizer=optimizer,
            loss=masked_sparse_categorical_crossentropy,
            metrics=[F1ScoreMetric()]  # Only use F1-score, remove accuracy
        )
    except TypeError:
        print("Older TensorFlow version detected, using decay parameter instead of weight_decay")
        optimizer = keras.optimizers.AdamW(
            learning_rate=3e-5,
            decay=weight_decay
        )
        model.compile(
            optimizer=optimizer,
            loss=masked_sparse_categorical_crossentropy,
            metrics=[F1ScoreMetric()]  # Only use F1-score, remove accuracy
        )
    
    return model

# Define the custom loss function outside of build_model() so it can be referenced when loading
def masked_sparse_categorical_crossentropy(y_true, y_pred):

    # Create mask for -100 values (0 where we have -100, 1 elsewhere)
    mask = tf.cast(tf.not_equal(y_true, -100), tf.float32)
    # Replace -100 values with 0 to avoid errors in loss computation
    y_true_safe = tf.where(tf.equal(y_true, -100), tf.zeros_like(y_true), y_true)
    # Compute loss for all positions
    loss_per_token = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_safe, y_pred, from_logits=True)
    # Apply mask to loss - multiply by 0 for masked positions
    masked_loss = loss_per_token * mask
    # Compute mean over valid positions only
    sum_loss = tf.reduce_sum(masked_loss)
    sum_mask = tf.reduce_sum(mask)
    # Avoid division by zero
    mean_loss = tf.cond(
        tf.equal(sum_mask, 0),
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: sum_loss / sum_mask
    )
    return mean_loss

# Build model
model = build_model()
print(model.summary())

# Update the model training section with early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=5,              # Number of epochs with no improvement after which training will stop
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1,               # Print messages when early stopping is triggered
    min_delta=0.001,         # Minimum change to qualify as an improvement
    mode='min'               # The direction is automatically inferred if not set, but we set it to be clear
)

# Model checkpoint to save best model
model_checkpoint = ModelCheckpoint(
    "best_model_weights.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# 5. Train the model with regularization
epochs = 50  # Increase max epochs since we have early stopping
batch_size = 21  # Adjust based on your data and memory

# Create model with increased dropout and weight decay
model = build_model(dropout_rate=0.5, weight_decay=0.02)


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-6)

# Train with callbacks
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    verbose=1
)

# Load the best weights after training
try:
    model.load_weights("best_model_weights.h5")
    print("Loaded best model weights from checkpoint")
except:
    print("Could not load best weights, using the final model")

# 6. Plot training history
# Update the plot_history function to remove accuracy plot
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'], label='Train')
    plt.plot(history.history['val_f1_score'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_history(history)

# 7. Evaluate and make predictions
def predict(model, texts):
    # Tokenize the input texts
    examples = []
    for text in texts:
        if isinstance(text, str):
            words = text.split()
        else:  # Assume it's already a list of words
            words = text
            
        # Tokenize each word and create inputs
        word_tokens = []
        word_token_map = []  # Map from token index to original word index
        
        for i, word in enumerate(words):
            # Use custom tokenization to keep ".", ":", "," attached
            # First create subwords using BERT tokenizer
            tokenized_word = tokenizer.tokenize(word)
            
            # Custom handling to merge tokens that got split due to special characters
            fixed_tokens = []
            current_token = ""
            for token in tokenized_word:
                # Check if this is a special character token we want to keep attached
                if token in [".", ",", ":"]:
                    # If we have accumulated tokens, add them to the fixed tokens list
                    if current_token:
                        fixed_tokens.append(current_token)
                        current_token = ""
                    # Add the special character as a separate token
                    fixed_tokens.append(token)
                else:
                    if not current_token:
                        current_token = token
                    else:
                        current_token += token.replace("##", "")
            
            # Add any remaining accumulated token
            if current_token:
                fixed_tokens.append(current_token)
                
            # Use our fixed tokens
            word_tokens.extend(fixed_tokens)
            word_token_map.extend([i] * len(fixed_tokens))
        
        # Add [CLS] and [SEP] tokens
        tokens = ['[CLS]'] + word_tokens + ['[SEP]']
        token_map = [-1] + word_token_map + [-1]  # -1 for special tokens
        
        # Convert to ids and create attention mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_map = token_map + ([-1] * padding_length)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_map = token_map[:max_length]
        
        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_map": token_map,
            "words": words
        })
    
    # Prepare inputs for the model
    batch_input_ids = np.array([ex["input_ids"] for ex in examples])
    batch_attention_mask = np.array([ex["attention_mask"] for ex in examples])
    
    # Make predictions
    logits = model.predict({
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask
    })
    
    predictions = np.argmax(logits, axis=-1)
    
    # Convert predictions to labels, accounting for subword tokens
    result = []
    for i, example in enumerate(examples):
        word_predictions = [None] * len(example["words"])
        token_map = example["token_map"]
        
        for j, token_pred in enumerate(predictions[i]):
            word_idx = token_map[j]
            if word_idx != -1:  # Skip special tokens and padding
                # If this is the first subword of the word, use its prediction
                if word_predictions[word_idx] is None:
                    word_predictions[word_idx] = id2label[token_pred]
        
        result.append(list(zip(example["words"], word_predictions)))
    
    return result

# Example usage
test_texts = ["Office Supplies 10 $5.99 $59.90",
            "Printer Paper 5 $12.50 $62.50",
            "Paper Clips 100 $2.99 $299.00"]

predictions = predict(model, test_texts)
for text_idx, pred in enumerate(predictions):
    print(f"Text: {test_texts[text_idx]}")
    for word, label in pred:
        print(f"{word}: {label}")
    print()

# 8. Save the model
model.save("invoice_bert_sequence_labeling")
tokenizer.save_pretrained("invoice_bert_tokenizer")
print("Model and tokenizer saved successfully!")

# 9. Function to process a new invoice
def process_invoice(model, tokenizer, invoice_text):
    """
    Process a new invoice and extract structured information
    
    Args:
        model: Trained TensorFlow model
        tokenizer: BERT tokenizer
        invoice_text: String containing invoice text
        
    Returns:
        Dictionary with extracted information
    """
    # Custom tokenization that preserves ".", ":", "," as part of words
    words = []
    current_word = ""
    
    # Split by whitespace first
    for token in invoice_text.split():
        # Apply custom splitting for special characters except ".", ":", ","
        special_char_pattern = r'[()\[\]{}\-;%$£€]'
        parts = re.split(f'({special_char_pattern})', token)
        parts = [p for p in parts if p]  # Remove empty strings
        
        # Add all non-empty parts as separate tokens
        for part in parts:
            if part:
                words.append(part)
    
    # Predict labels using our updated words list
    predictions = predict(model, [words])[0]
    
    # Extract entities
    entities = {}
    current_entity = None
    current_value = []
    
    for word, label in predictions:
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
                        entities[current_entity] = entity_value
                else:
                    entities[current_entity] = entity_value
                
                current_entity = None
                current_value = []
    
    # Save the last entity if we were building one
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

# Example of processing an invoice
sample_invoice = """
Office Supplies 10 $5.99 $59.90
Printer Paper 5 $12.50 $62.50
Ink Cartridges 3 $35.00 $105.00
Total Amount: $227.40
Tax Rate: 8%
Tax: $16.80
"""

# Add a function to load a trained model
# Update load_trained_model to include custom metric
def load_trained_model(model_path="invoice_bert_sequence_labeling", tokenizer_path="invoice_bert_tokenizer"):
    """
    Load a previously saved model and tokenizer
    
    Args:
        model_path: Path to the saved model
        tokenizer_path: Path to the saved tokenizer
        
    Returns:
        model: The loaded TensorFlow model
        tokenizer: The loaded tokenizer
    """
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(
            model_path, 
            # Custom loss function and metrics need to be explicitly provided
            custom_objects={
                "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
                "F1ScoreMetric": F1ScoreMetric
            }
        )
        print("Model loaded successfully")
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

# Update the evaluate_model function to fix the classification report issue
def evaluate_model(model, val_dataset):
    """
    Evaluate the model on the validation dataset and generate a classification report
    
    Args:
        model: Trained TensorFlow model
        val_dataset: Validation dataset
        
    Returns:
        Tuple of F1-scores (micro, macro, weighted)
    """
    # Get validation predictions
    all_true_labels = []
    all_pred_labels = []
    
    for batch in val_dataset:
        inputs, labels = batch
        
        # Get model predictions
        logits = model.predict(inputs)
        predictions = np.argmax(logits, axis=-1)
        
        # Process each sequence in the batch
        for i in range(len(predictions)):
            true_seq = labels[i].numpy()
            pred_seq = predictions[i]
            
            # Filter out padding and special tokens (-100)
            mask = true_seq != -100
            true_labels = true_seq[mask]
            pred_labels = pred_seq[mask]
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
    
    # Convert numeric labels to text labels
    true_labels_text = [id2label[label] for label in all_true_labels]
    pred_labels_text = [id2label[label] for label in all_pred_labels]
    
    # Get all unique labels that actually appear in the data
    unique_label_ids = sorted(list(set(all_true_labels)))
    unique_label_names = [id2label[id] for id in unique_label_ids]
    
    # Calculate F1 score
    micro_f1 = f1_score(all_true_labels, all_pred_labels, average='micro')
    macro_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    weighted_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    
    print("\n===== Model Evaluation =====")
    print(f"Micro F1-score: {micro_f1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # Generate classification report with numeric labels (using only labels that appear in the data)
    print("\n===== Classification Report (Numeric Labels) =====")
    try:
        print(classification_report(
            all_true_labels, all_pred_labels,
            labels=unique_label_ids,
            target_names=unique_label_names,
            digits=4,
            zero_division=0
        ))
    except ValueError as e:
        print(f"Could not generate numeric label report: {e}")
        # Fallback to simpler classification report without target_names
        print(classification_report(
            all_true_labels, all_pred_labels,
            digits=4,
            zero_division=0
        ))
    
    # Get unique text labels that appear in the data
    unique_text_labels = sorted(list(set(true_labels_text)))
    
    # Generate text-based classification report (using only labels that appear in the data)
    print("\n===== Classification Report (Text Labels) =====")
    try:
        print(classification_report(
            true_labels_text, pred_labels_text,
            labels=unique_text_labels,
            digits=4,
            zero_division=0
        ))
    except ValueError as e:
        print(f"Could not generate text label report: {e}")
        # Fallback to basic report without specifying labels
        print(classification_report(
            true_labels_text, pred_labels_text,
            digits=4,
            zero_division=0
        ))
    
    return micro_f1, macro_f1, weighted_f1

# Add a function to evaluate the model and generate a classification report
def evaluate_model(model, val_dataset):
    """
    Evaluate the model on the validation dataset and generate a classification report
    
    Args:
        model: Trained TensorFlow model
        val_dataset: Validation dataset
        
    Returns:
        None (prints evaluation metrics)
    """
    # Get validation predictions
    all_true_labels = []
    all_pred_labels = []
    
    for batch in val_dataset:
        inputs, labels = batch
        
        # Get model predictions
        logits = model.predict(inputs)
        predictions = np.argmax(logits, axis=-1)
        
        # Process each sequence in the batch
        for i in range(len(predictions)):
            true_seq = labels[i].numpy()
            pred_seq = predictions[i]
            
            # Filter out padding and special tokens (-100)
            mask = true_seq != -100
            true_labels = true_seq[mask]
            pred_labels = pred_seq[mask]
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
    
    # Convert numeric labels to text labels
    true_labels_text = [id2label[label] for label in all_true_labels]
    pred_labels_text = [id2label[label] for label in all_pred_labels]
    
    # Extract unique labels for classification report
    unique_labels = list(set([label for label in true_labels_text if not label.startswith("I-")]))
    
    # Calculate F1 score
    micro_f1 = f1_score(all_true_labels, all_pred_labels, average='micro')
    macro_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    weighted_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    
    print("\n===== Model Evaluation =====")
    print(f"Micro F1-score: {micro_f1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # Generate classification report
    print("\n===== Classification Report =====")
    print(classification_report(
        all_true_labels, all_pred_labels,
        target_names=[id2label[i] for i in range(len(labels))],
        digits=4
    ))
    
    # Generate text-based classification report
    print("\n===== Classification Report (Text Labels) =====")
    print(classification_report(
        true_labels_text, pred_labels_text,
        labels=unique_labels,
        digits=4
    ))
    
    # Return F1-score for potential further use
    return micro_f1, macro_f1, weighted_f1

# Add a main block for testing a saved model on sample invoice
# Update the __main__ block to include model evaluation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test invoice NER model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "evaluate"],
                        help="Whether to train a new model or test/evaluate an existing one")
    parser.add_argument("--model_path", type=str, default="invoice_bert_sequence_labeling",
                        help="Path to saved model when in test/evaluate mode")
    parser.add_argument("--tokenizer_path", type=str, default="invoice_bert_tokenizer",
                        help="Path to saved tokenizer when in test/evaluate mode")
    parser.add_argument("--invoice", type=str, default=None,
                        help="Sample invoice text to process in test mode (uses default if not provided)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # After training, evaluate the model
        print("\nEvaluating model on validation data...")
        evaluate_model(model, val_dataset)
    
    elif args.mode == "evaluate":
        # Load a trained model and evaluate it
        model, tokenizer = load_trained_model(args.model_path, args.tokenizer_path)
        
        if model is None or tokenizer is None:
            print("Failed to load model or tokenizer. Exiting.")
            exit(1)
        
        # Create validation dataset from the trained data
        print("Creating validation dataset...")
        val_dataset = create_tf_dataset(val_inputs, val_labels, batch_size)
        
        # Evaluate model and print classification report
        evaluate_model(model, val_dataset)
    
    elif args.mode == "test":
        # Load a trained model and test it on sample invoice
        model, tokenizer = load_trained_model(args.model_path, args.tokenizer_path)
        
        if model is None or tokenizer is None:
            print("Failed to load model or tokenizer. Exiting.")
            exit(1)
        
        # Process either the provided invoice or the sample one
        invoice_text = args.invoice if args.invoice else sample_invoice
        
        print("\n" + "="*50)
        print("Sample Invoice Text:")
        print(invoice_text)
        print("="*50 + "\n")
        
        # Extract structured data
        extracted_data = process_invoice(model, tokenizer, invoice_text)
        print("Extracted Invoice Data:")
        for key, value in extracted_data.items():
            print(f"{key}: {value}")
        
        # You can also use the predict function to see token-level predictions
        predictions = predict(model, [invoice_text])
        print("\nToken-level predictions:")
        for word, label in predictions[0]:
            print(f"{word}: {label}")