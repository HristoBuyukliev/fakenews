import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data = pd.read_csv('data/FN_Training_Set.csv', encoding='cp1251')
target_cols = ['fake_news_score', 'click_bait_score']

stopwords = []
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='cp1251', 
        ngram_range=(1,5), 
        min_df=10)

X = vectorizer.fit_transform(data.Content.values.astype('U'))

