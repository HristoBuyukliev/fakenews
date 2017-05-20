import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/FN_Training_Set.csv', encoding='cp1251')
target_cols = ['fake_news_score', 'click_bait_score']

stopwords = []
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='cp1251', 
        ngram_range=(1,5), 
        min_df=100)

X = vectorizer.fit_transform(data.Content.values.astype('U'))
y = data.click_bait_score.values
y = preprocessing.label_binarize(data.click_bait_score.values, classes = [0,1,2,3])
lr = LogisticRegression()


