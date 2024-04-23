import pandas as pd
import nltk
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')
print(1)
# Load dataset
df = pd.read_csv('C:\\Users\\death\\Desktop\\minor project codes\\combined_data.csv')

# Function to preprocess tweets
def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # Replace mentions (@user)
    tweet = re.sub(r'@[^\s]+', 'USER', tweet)
    # Remove continuously repeated symbols or characters
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet
print(2)
# Apply preprocessing to tweets
df['clean_tweet'] = df['tweet'].apply(preprocess_tweet)

# Tokenization, Lemmatization, and Stopwords removal
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print(3)
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]  # Remove stopwords and single characters
    return tokens
print(4)
# Apply tokenization and lemmatization
df['tokenized_tweet'] = df['clean_tweet'].apply(tokenize_and_lemmatize)

# Convert tokenized tweets back to strings
df['clean_tweet_processed'] = df['tokenized_tweet'].apply(lambda x: ' '.join(x))
print(5)
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
tfidf_features = tfidf_vectorizer.fit_transform(df['clean_tweet_processed'])
print(6)
# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
