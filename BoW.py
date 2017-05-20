import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
TRAIN_SPLIT = 2300
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
y = data.fake_news_score.values

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# logistic regression
for C in [10**(i/2) for i in range(-6,7)]:
	print C
	lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C)
	lr.fit(x_train, y_train)
	predictions = lr.predict(x_test)
	print metrics.accuracy_score(predictions, y_test)



# SVMs
for kernel in ['linear', 'poly', 'rbf']:
	for C in [1000, 300, 100, 30, 10, 3, 1, 0.3, 0.1]:
		print C
		print kernel
		svm = SVC(kernel=kernel, C=C)
		svm.fit(X[:TRAIN_SPLIT], y[:TRAIN_SPLIT])
		predictions = svm.predict(X[TRAIN_SPLIT:])
		print metrics.accuracy_score(predictions, y[TRAIN_SPLIT:])

# Linear regression
lr = LinearRegression()
lr.fit(X[:TRAIN_SPLIT], y[:TRAIN_SPLIT])
predictions = lr.predict(X[TRAIN_SPLIT:]).round().clip(0,3)
print metrics.accuracy_score(predictions, y[TRAIN_SPLIT:])

# decision tree
for depth in [2, 5, 10]:
	dt = DecisionTreeClassifier(max_depth=depth)
	dt.fit(x_train, y_train)
	predictions = dt.transform(x_test)
	print metrics.accuracy_score(y_test, predictions)