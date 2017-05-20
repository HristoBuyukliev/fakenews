import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

TRAIN_SPLIT = 2000
N_GRAM_RANGE = 5

data = pd.read_csv('data/FN_Training_Set.csv', encoding='cp1251')
target_cols = ['fake_news_score', 'click_bait_score']


stopwords = []
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='cp1251', 
        ngram_range=(1,N_GRAM_RANGE), 
        min_df=100)

X = vectorizer.fit_transform(data.Content.values.astype('U'))
y = data.click_bait_score.values
# y = y.reshape(-1,1)
# y = preprocessing.label_binarize(data.click_bait_score.values, classes = [0,1,2,3])
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

lr.fit(X, y)
cross_val_score(lr, X, y, 'accuracy')

