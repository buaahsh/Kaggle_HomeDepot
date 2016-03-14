import math
import time


start_time = time.time()


class BM25(object):
    """docstring for BM25"""
    def __init__(self, modelFile):
        super(BM25, self).__init__()
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = 0
        self.idf = {}
        self.N = 0

    def __load(self):
        pass

    def getNq(self, q, docs):
        if q in docs:
            return docs[q]
        return 0

    def buildDocs(self, docs):
        _dict = {}
        for d in docs:
            for i in set(d.split(' ')):
                if i not in _dict:
                    _dict[i] = 0
                _dict[i] += 1
        return _dict

    def build(self, querySet, docs):
        N = len(docs)
        print "DOCS Length : ", N
        self.N = N
        self.avgdl = sum(len(d.split(' ')) for d in docs) * 1.0 / N
        print("--- Count avgdl: %s minutes ---" %
            round(((time.time() - start_time) / 60), 2))

        docs = self.buildDocs(docs)

        print("--- Build docs: %s minutes ---" %
          round(((time.time() - start_time) / 60), 2))

        for q in querySet:
            for i in q.split():
                if i not in self.idf:                
                    nq = self.getNq(q, docs)
                    idf = math.log((N - nq + 0.5) / (nq + 0.5))
                    self.idf[q] = idf

        print("--- IDF build: %s minutes ---" %
          round(((time.time() - start_time) / 60), 2))

    def str_common_word(self, str1, str2):
        return sum(int(str2.find(word) >= 0) for word in str1.split())

    def getIDF(self, w):
        if w in self.idf:
            return self.idf[w]
        else:
            return math.log((self.N + 0.5) / (0.5))

    def score(self, q, d):
        D = len(d.split(' '))
        tf = self.str_common_word(q, d)
        score = sum(self.getIDF(i) * (tf * (self.k1 + 1) 
            / (tf + self.k1 * (1 - self.b + self.b * D / self.avgdl))) for i in q.split())
        return score

def Train():
    import numpy as np
    import pandas as pd
    import rd
    df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
    # df_attr = pd.read_csv('../input/attributes.csv')
    df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
    print("--- Files Loaded: %s minutes ---" %
          round(((time.time() - start_time) / 60), 2))
    num_train = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all['search_term'] = df_all['search_term'].map(lambda x: rd.str_stemmer(x))
    df_all['product_title'] = df_all[
        'product_title'].map(lambda x: rd.str_stemmer(x))
    df_all['product_description'] = df_all[
        'product_description'].map(lambda x: rd.str_stemmer(x))
    print("--- Stem: %s minutes ---" %
          round(((time.time() - start_time) / 60), 2))
    bm25 = BM25(None)
    bm25.build(df_all['search_term'], df_all['product_description'])
    import cPickle as p  
    f = file("bm25.m", 'w')  
    p.dump(bm25, f) # dump the object to a file  
    f.close()  

if __name__ == "__main__":
    Train()



