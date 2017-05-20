import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


all_data = pd.read_csv('data/main_data_fake_news.csv')
all_data.content.fillna(' ', inplace=True)
corpus = all_data.groupby('host').aggregate(lambda x: ' '.join(x.content))
stopwords = []
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        ngram_range=(1,1), 
        min_df=100)

tsne = TSNE(metric='cosine')
embedded = tsne.fit_transform(X.toarray())

