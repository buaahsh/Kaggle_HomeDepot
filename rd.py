import numpy as np
import pandas as pd
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import stopwords
import time

# words = stopwords.words('english')
# words = []
stemmer = SnowballStemmer('english')


def stopwords_filter(w):
    if w in words:
        return False
    return True


def str_stemmer(s):
    # if isinstance(s, str) == False:
    #     s = str(s)
    # return s
    if type(s) == float:
        s = str(s)
    _list = [stemmer.stem(word) for word in s.lower().split()]
    # _list = filter(stopwords_filter, _list)
    return " ".join(_list)


import cPickle as p
from bm25 import BM25
f = file('bm25.m')  
bm25 = p.load(f)
print "IDF Length: ", len(bm25.idf)

def bm25_score(str1, str2):
    return bm25.score(str1, str2)

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_

if __name__ == "__main__":
    start_time = time.time()

    df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
    # df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
    # # df_attr = pd.read_csv('../input/attributes.csv')
    # df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
    # df_attr = pd.read_csv('./input/attributes.csv', encoding="ISO-8859-1")
    # # df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid",
    # #                                                       "value"]].rename(columns={"value": "brand"})

    num_train = df_train.shape[0]
    # df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    # df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    # # df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    # print("--- Files Loaded: %s minutes ---" %
    #       round(((time.time() - start_time) / 60), 2))

    # df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
    # df_all['product_title'] = df_all[
    #     'product_title'].map(lambda x: str_stemmer(x))
    # df_all['product_description'] = df_all[
    #     'product_description'].map(lambda x: str_stemmer(x))
    # # df_all['brand'] = df_all['brand'].map(lambda x: str_stemmer(x))

    # print("--- Stemmed: %s minutes ---" %
    #       round(((time.time() - start_time) / 60), 2))

    # df_all['len_of_query'] = df_all['search_term'].map(
    #     lambda x: len(x.split())).astype(np.int64)

    # df_all['product_info'] = df_all['search_term'] + "\t" + \
    #     df_all['product_title'] + "\t" + df_all['product_description']# + "\t" + df_all['brand']

    # df_all['word_in_title'] = df_all['product_info'].map(
    #     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
    # df_all['word_in_description'] = df_all['product_info'].map(
    #     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

    # print("--- Query In: %s minutes ---" %
    #       round(((time.time() - start_time) / 60), 2))

    # df_all['query_last_word_in_title'] = df_all['product_info'].map(
    #     lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
    # df_all['query_last_word_in_description'] = df_all['product_info'].map(
    #     lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
    # print("--- Query Last Word In: %s minutes ---" %
    #       round(((time.time() - start_time) / 60), 2))
    
    # df_all['bm25_in_description'] = df_all['product_info'].map(
    #     lambda x: bm25_score(x.split('\t')[0], x.split('\t')[2]))
    # # df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
    # # df_all['word_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
    # # df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
    # # df_brand = pd.unique(df_all.brand.ravel())
    # # d={}
    # # i = 1000
    # # for s in df_brand:
    # #     d[s]=i
    # #     i+=3
    # # df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
    # # df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))
    
    # df_all = df_all.drop(
    #     ['search_term', 'product_title', 'product_description', 'product_info'], axis=1)

    # df_all.to_csv('df_all.csv')
    df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']

    y_train = df_train['relevance'].values
    X_train = df_train.drop(['id', 'relevance'], axis=1).values
    X_test = df_test.drop(['id', 'relevance'], axis=1).values


    RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=0) # 15 6
    clf = BaggingRegressor(
        rf, n_estimators=45, max_samples=0.1, random_state=25)

    clf.fit(X_train, y_train)
    # param_grid = {}
    # model = grid_search.GridSearchCV(
    #     estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
    # model.fit(X_train, y_train)

    # print("Best parameters found by grid search:")
    # print(model.best_params_)
    # print("Best CV score:")
    # print(model.best_score_)

    # y_pred = model.predict(X_test)

    y_pred = clf.predict(X_test)

    print("--- Model Trained: %s minutes ---" %
          round(((time.time() - start_time) / 60), 2))

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(
        'submission.csv', index=False)
