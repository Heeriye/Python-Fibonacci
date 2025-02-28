# Importing dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.optimizers import Adam

# Load dataset (Example: Shakespeare Text)
file_path = 'shakespeare.txt'  # Provide the path to your text dataset

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read().lower()  # Convert text to lowercase

# Remove special characters
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Tokenizing text
nltk.download('punkt')
tokens = word_tokenize(text)

# Create sequences
input_sequences = []
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)
total_words = len(tokenizer.word_index) + 1

for i in range(1, len(tokens)):
    n_gram_sequence = tokens[:i+1]
    input_sequences.append(tokenizer.texts_to_sequences([' '.join(n_gram_sequence)])[0])

# Pad sequences
max_seq_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

# Split data into features and labels
X = input_sequences[:, :-1]  # Features (input words)
y = input_sequences[:, -1]   # Labels (next word)
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build LSTM Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_seq_length-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, verbose=1)

# Text Generation Function
def generate_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate new text
seed_text = "to be or not to be"
generated_text = generate_text(seed_text, next_words=20)
print("\nGenerated Text:\n", generated_text)
