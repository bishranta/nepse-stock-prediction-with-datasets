import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ✅ Load labeled dataset
csv_path = 'nepse-news-labeled.csv'
df = pd.read_csv(csv_path)

# ✅ Map sentiments to numerical labels
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['top_sentiment'].map(sentiment_mapping)

# ✅ Data Augmentation: duplicate negative samples
negative_samples = df[df['top_sentiment'] == 'negative']
df = pd.concat([df, negative_samples] * 10, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ Compute class weights
labels_numeric = df['sentiment_label'].values
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_numeric), y=labels_numeric)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ✅ Tokenization and Padding
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
GLOVE_PATH = 'glove.6B.100d.txt'

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(df['title'])
X = tokenizer.texts_to_sequences(df['title'])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
y = tf.keras.utils.to_categorical(df['sentiment_label'], num_classes=3)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Load GloVe embeddings
embeddings_index = {}
with open(GLOVE_PATH, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coeffs

word_index = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ✅ Define CNN Model
model = Sequential([
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# ✅ Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Train Model with class weights
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# ✅ Save Model
model.save('sentiment_cnn_model.keras')

# ✅ Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# ✅ Predict sentiments and export to CSV matching the HTML
sentiment_classes = ['negative', 'neutral', 'positive']

predictions = model.predict(X)
predicted_sentiments = [sentiment_classes[np.argmax(pred)] for pred in predictions]
sentiment_scores = [round(pred[2] - pred[0], 4) for pred in predictions]

result_df = df.copy()
result_df['predicted_sentiment'] = predicted_sentiments
result_df['sentiment_score'] = sentiment_scores

# ✅ Reorder and save CSV to match the HTML expectation
result_df[['datetime', 'title', 'source', 'link', 'predicted_sentiment', 'sentiment_score']].to_csv('sentiment_results_full.csv', index=False)
print("Predicted sentiments with all original columns saved to 'sentiment_results_full.csv'")
