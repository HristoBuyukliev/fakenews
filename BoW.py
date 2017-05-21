import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier


N_GRAM_RANGE = 5

data = pd.read_csv('data/FN_Training_Set.csv', encoding='cp1251')
target_cols = ['fake_news_score', 'click_bait_score']


with open('stopwords-bg.txt') as rfile:
    stopwords = rfile.readlines()
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='cp1251', 
        ngram_range=(1,N_GRAM_RANGE), 
        min_df=100)

X = vectorizer.fit_transform(data.Content.values.astype('U'))
y = data.fake_news_score.values

# shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# # logistic regression
# for C in [10**(i/2.) for i in range(-6,7)]:
# 	print C
# 	lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C)
# 	lr.fit(x_train, y_train)
# 	predictions = lr.predict(x_test)
# 	print metrics.accuracy_score(predictions, y_test)



# # SVMs
# for kernel in ['linear', 'rbf']:
# 	for C in [1000, 100, 30, 10, 3, 1]:
# 		print C
# 		print kernel
# 		svm = SVC(kernel=kernel, C=C)
# 		svm.fit(x_train, y_train)
# 		predictions = svm.predict(x_test)
# 		print metrics.accuracy_score(predictions, y_test)

# # Linear regression
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# predictions = lr.predict(x_test).round().clip(0,3)
# print metrics.accuracy_score(predictions, y_test)

# decision tree
for depth in [2, 5, 10]:
		depth = 5
		print depth
		dt = DecisionTreeClassifier(max_depth=depth)
		dt.fit(x_train, y_train)
		predictions = dt.predict(x_test)
		print metrics.accuracy_score(y_test, predictions)

# # random forests
# for depth in [2, 5, 10]:
# 	for n_trees in [10, 20, 50, 100, 200, 500, 1000]:
# 		print depth, n_trees
# 		rf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
# 		rf.fit(x_train, y_train)
# 		predictions = rf.predict(x_test)
# 		print metrics.accuracy_score(y_test, predictions)

# for depth in [10, 20]:
# 	for n_trees in [1000, 5000]:
# 		print depth, n_trees
# 		xgb = xgboost.XGBClassifier(max_depth=depth, n_estimators=n_trees)
# 		xgb.fit(x_train, y_train)
# 		predictions = xgb.predict(x_test)
# 		print metrics.accuracy_score(y_test, predictions)

# Ensemble
lr  = LogisticRegression(C=100)
xgb = xgboost.XGBClassifier(max_depth=20, n_estimators=1000)
svm = SVC(kernel='linear', C=10)
ensemble = VotingClassifier(estimators = [('lr', lr), ('xgb', xgb), ('svm', svm)])
for clf, label in zip([lr, xgb, svm, ensemble], ['Logistic Regression', 'xgboost', 'SVM', 'Ensemble']):
	scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))