# -*- coding: utf-8 -*-
"""MBC Week 4 F - Fixed Version for Streamlit Cloud"""

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess function
def preprocess(text):
    new_text = []
    for t in str(text).split(" "):  # Ensure text is string
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        t = t.replace("#", "")
        new_text.append(t.lower())
    return " ".join(new_text).strip().replace("  ", " ")

# Apply preprocessing
train_df['text'] = train_df['text'].apply(preprocess)
test_df['text'] = test_df['text'].apply(preprocess)

# Label mapping
label_order = ['SADNESS', 'ANGER', 'SUPPORT', 'HOPE', 'DISAPPOINTMENT']
label_map = {label: i for i, label in enumerate(label_order)}
id_to_label = {i: label for label, i in label_map.items()}

train_df['label_encoded'] = train_df['label'].map(label_map)

# Sequence length
SEQ_LEN = 105

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=".cache")

# Encode training data
train_encoded_inputs = tokenizer(
    train_df['text'].tolist(),
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=SEQ_LEN,
    return_token_type_ids=False,
    return_tensors='tf'
)

# TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_encoded_inputs, train_df['label_encoded'].values))

def map_bert(inputs, labels):
    inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    return inputs, labels

train_dataset = train_dataset.map(map_bert)

# Train/val split
DS_LEN = len(train_dataset)
SPLIT = 0.8
train_ds = train_dataset.take(round(DS_LEN * SPLIT)).batch(64)
val_ds = train_dataset.skip(round(DS_LEN * SPLIT)).batch(64)

# Model
n_classes = len(label_order)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Training
print("Starting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,  # reduce epochs for demo
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
)
print("Training finished.")

# Encode test data
test_encoded_inputs = tokenizer(
    test_df['text'].tolist(),
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=SEQ_LEN,
    return_token_type_ids=False,
    return_tensors='tf'
)

test_dataset = tf.data.Dataset.from_tensor_slices(test_encoded_inputs)
test_dataset = test_dataset.map(lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}).batch(64)

# Predictions
print("Making predictions on test data...")
predictions = model.predict(test_dataset)
predicted_indices = np.argmax(predictions.logits, axis=1)

try:
    test_labels_df = pd.read_csv('test_labels.csv')
    test_labels_df['label_encoded'] = test_labels_df['label'].map(label_map)
    y_true = test_labels_df['label_encoded'].values

    f1 = f1_score(y_true, predicted_indices, average='weighted')
    print(f"\nFinal Weighted F1 Score: {f1:.4f}")

except FileNotFoundError:
    print("\nWarning: Could not find 'test_labels.csv'. F1 score cannot be calculated.")

# Save submission
submission_df = pd.DataFrame({
    'id_comment': test_df['id_comment'],
    'label': predicted_indices
})
submission_df.to_csv('submission2.csv', index=False)
print("\nSubmission file 'submission2.csv' created successfully!")
