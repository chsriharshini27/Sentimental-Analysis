# Sentimental-Analysis


Theory of Sentiment Analysis

Text Preprocessing:

Tokenization: Splitting text into individual words or tokens.
Lowercasing: Converting all text to lowercase to maintain consistency.
Removing Punctuation: Stripping punctuation marks from the text.
Stop Words Removal: Removing common words (like "the", "is", "in") that do not contribute to sentiment.
Stemming/Lemmatization: Reducing words to their root form (e.g., "running" to "run").

Feature Extraction:

Bag of Words (BoW): Representing text as a collection of word counts.
TF-IDF (Term Frequency-Inverse Document Frequency): Weighing words based on their importance.
Word Embeddings: Using pre-trained models like Word2Vec or GloVe to represent words as vectors.

Model Selection:

Machine Learning Models: Logistic Regression, Naive Bayes, SVM, etc.
Deep Learning Models: LSTM, GRU, BERT, etc.

Training and Evaluation:

Splitting Data: Dividing the dataset into training and testing sets.
Model Training: Training the model on the training data.
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, etc.

Deployment:

Saving the Model: Using libraries like joblib or pickle.
API Creation: Using Flask or FastAPI to create an API for your model.

Example code:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')

# Sample data
data = {
    'text': ['I love this product', 'This is the worst thing ever', 'Absolutely fantastic!', 'Not good at all'],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Text preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
df['text'] = df['text'].apply(preprocess)

# Features and labels
X = df['text']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
