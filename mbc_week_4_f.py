import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF logs

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

st.set_page_config(page_title="MBC Week 4 - BERT (TF)", layout="wide")

# ---------- Utils ----------
def preprocess(text: str) -> str:
    new_text = []
    for t in str(text).split(" "):
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        t = t.replace("#","")
        new_text.append(t.lower())
    return " ".join(new_text).strip().replace("  ", " ")

label_order = ['SADNESS', 'ANGER', 'SUPPORT', 'HOPE', 'DISAPPOINTMENT']
label_map = {label: i for i, label in enumerate(label_order)}
id_to_label = {i: label for label, i in label_map.items()}

SEQ_LEN = 96
BATCH = 16
EPOCHS = 1  # naikin kalau perlu

# ---------- UI ----------
st.title("MBC Week 4 — BERT Sequence Classification (TensorFlow)")
st.write("Pastikan file **train.csv** dan **test.csv** ada di repo/app directory. Atau upload di bawah.")

train_file = st.file_uploader("Upload train.csv (opsional, kalau tidak ada akan dibaca dari disk)", type=["csv"], key="train")
test_file  = st.file_uploader("Upload test.csv  (opsional, kalau tidak ada akan dibaca dari disk)",  type=["csv"], key="test")

# ---------- Load data ----------
def load_csv(fp: str, uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(fp)

@st.cache_data(show_spinner=False)
def load_data(train_uploaded, test_uploaded):
    train_df = load_csv("train.csv", train_uploaded)
    test_df  = load_csv("test.csv",  test_uploaded)
    # expected columns: train: ['id_comment','text','label']; test: ['id_comment','text']
    train_df['text'] = train_df['text'].apply(preprocess)
    test_df['text']  = test_df['text'].apply(preprocess)
    train_df['label_encoded'] = train_df['label'].map(label_map)
    return train_df, test_df

try:
    train_df, test_df = load_data(train_file, test_file)
    st.success(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
except Exception as e:
    st.error(f"Gagal baca CSV: {e}")
    st.stop()

# ---------- Tokenizer ----------
@st.cache_resource(show_spinner=True)
def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenizer = get_tokenizer()

def tokenize_texts(texts):
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=SEQ_LEN,
        return_token_type_ids=False,
        return_tensors='tf'
    )

# ---------- Train ----------
def make_tf_dataset(encodings, labels=None, batch_size=BATCH, shuffle=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
        })
    else:
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
            },
            tf.convert_to_tensor(labels, dtype=tf.int32)
        ))
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

@st.cache_resource(show_spinner=True)
def build_model(num_labels: int):
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_labels
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model

col1, col2 = st.columns(2)
with col1:
    st.write("Sample data (train):")
    st.dataframe(train_df.head(5))
with col2:
    st.write("Label distribution:")
    st.bar_chart(train_df['label'].value_counts())

if st.button("Train & Predict"):
    with st.spinner("Tokenizing & splitting…"):
        X_train, X_val, y_train, y_val = train_test_split(
            train_df['text'].tolist(),
            train_df['label_encoded'].values,
            test_size=0.2,
            random_state=42,
            stratify=train_df['label_encoded'].values
        )

        train_enc = tokenize_texts(X_train)
        val_enc   = tokenize_texts(X_val)

        train_ds = make_tf_dataset(train_enc, y_train, shuffle=True)
        val_ds   = make_tf_dataset(val_enc,   y_val)

    with st.spinner("Building & training model…"):
        model = build_model(num_labels=len(label_order))
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks
        )
        st.success("Training finished.")

    # Optional quick F1 on validation
    with st.spinner("Evaluating on validation set…"):
        val_logits = model.predict(val_ds).logits
        y_val_pred = np.argmax(val_logits, axis=1)
        try:
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            st.write(f"**Weighted F1 (val)**: {f1:.4f}")
        except Exception:
            pass

    # Predict test
    with st.spinner("Predicting test…"):
        test_enc = tokenize_texts(test_df['text'].tolist())
        test_ds  = make_tf_dataset(test_enc)
        test_logits = model.predict(test_ds).logits
        test_pred_idx = np.argmax(test_logits, axis=1)

        submission = pd.DataFrame({
            'id_comment': test_df['id_comment'],
            'label': test_pred_idx
        })
        out_path = "submission.csv"
        submission.to_csv(out_path, index=False)

    st.success("Done! File submission.csv siap diunduh.")
    st.download_button("Download submission.csv", data=submission.to_csv(index=False), file_name="submission.csv", mime="text/csv")
else:
    st.info("Klik **Train & Predict** untuk mulai.")
