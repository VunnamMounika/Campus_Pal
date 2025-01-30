import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import json

# Download the required NLTK data
nltk.download("punkt")  # For tokenization
nltk.download("punkt_tab")  # For `punkt_tab` support


# Custom implementation for CountVectorizer (feature extraction)
class SimpleVectorizer:
    def __init__(self):
        self.vocab = {}

    def fit_transform(self, corpus):
        self.vocab = {}
        bow_matrix = []
        for sentence in corpus:
            bow = {}
            for word in sentence.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                bow[self.vocab[word]] = bow.get(self.vocab[word], 0) + 1
            row = [bow.get(idx, 0) for idx in range(len(self.vocab))]
            bow_matrix.append(row)
        return np.array(bow_matrix)

    def transform(self, corpus):
        bow_matrix = []
        for sentence in corpus:
            bow = {}
            for word in sentence.split():
                if word in self.vocab:
                    index = self.vocab[word]
                    bow[index] = bow.get(index, 0) + 1
            row = [bow.get(idx, 0) for idx in range(len(self.vocab))]
            bow_matrix.append(row)
        return np.array(bow_matrix)


class SimpleVectorizer:
    def __init__(self):
        self.vocab = {}

    def fit_transform(self, corpus):
        self.vocab = {}

        # Build vocabulary with unique words
        for sentence in corpus:
            for word in sentence.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        # Create the BoW representation as fixed-length rows
        bow_matrix = []
        for sentence in corpus:
            bow = [0] * len(self.vocab)  # Initialize vector with zeros
            for word in sentence.split():
                if word in self.vocab:
                    index = self.vocab[word]
                    bow[index] += 1  # Increment count at the word's index
            bow_matrix.append(bow)

        return np.array(bow_matrix)  # Fixed-length rows

    def transform(self, corpus):
        bow_matrix = []
        for sentence in corpus:
            bow = [0] * len(self.vocab)  # Initialize vector with zeros
            for word in sentence.split():
                if word in self.vocab:
                    index = self.vocab[word]
                    bow[index] += 1  # Increment count at the word's index
            bow_matrix.append(bow)

        return np.array(bow_matrix)
class SimpleLabelEncoder:
    def __init__(self):
        # Dictionary to map class labels to integers.
        self.classes = {}

    def fit_transform(self, labels):
        # Maps each unique label to an integer and encodes them.
        self.classes = {}
        encoded_labels = []
        for label in labels:
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            encoded_labels.append(self.classes[label])
        return np.array(encoded_labels)

    def inverse_transform(self, encoded_labels):
        # Converts integer-encoded classes back to their original string labels.
        inv_classes = {v: k for k, v in self.classes.items()}
        return [inv_classes[label] for label in encoded_labels]



# Train the chatbot
def train_chatbot():
    # Example training data
    with open("C:/Users/bhara/PycharmProjects/Tensorflow-Chatbot/Bot/content.json", "r", encoding="utf-8") as file:
        dataFile = json.load(file)
    intents = dataFile  # Access the 'intents' list

    inputs, outputs = [], []
    for item in intents:
        question = item['question'].lower()  # Convert to lowercase
        tokenized_input = word_tokenize(question)  # Tokenize the sentence
        inputs.append(" ".join(tokenized_input))  # Store as a cleaned string
        outputs.append(item['answer'])  # Store answer

    # Feature extraction
    vectorizer = SimpleVectorizer()
    X = vectorizer.fit_transform(inputs)

    # One-hot encode the classes
    encoder = SimpleLabelEncoder()
    y_encoded = encoder.fit_transform(outputs)
    y = tf.keras.utils.to_categorical(y_encoded, num_classes=len(set(outputs)))

    # Build the model
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Input layer with feature dimension
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(y.shape[1], activation="softmax")  # Output classes
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X, y, epochs=25, batch_size=8)

    # Save model and preprocessor
    model.save("chatbot_model.h5")
    return vectorizer, encoder


# Predict using a pre-trained chatbot model
def predict_class(user_input, vectorizer, encoder):
    try:
        # Tokenize the input and preprocess
        tokenized_input = " ".join(word_tokenize(user_input.lower()))
        vectorized_input = vectorizer.transform([tokenized_input])

        # Load the pre-trained model
        model = load_model("chatbot_model.h5")
        probabilities = model.predict(vectorized_input)
        predicted_class = np.argmax(probabilities)
        return encoder.inverse_transform([predicted_class])[0]
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I didn't understand that."
