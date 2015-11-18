import os, json, datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from models.mSDA import *
from util.strutil import *
from testBase import clusterTestBase

__author__ = 'PauKung'

stop = stopwords.words('english')

class articleTest(clusterTestBase):

    def __init__(self, model):
        self.vectorizer = TfidfVectorizer(stop_words=stop, ngram_range=(1,2), min_df=2)
        self.model = model

    def load_data(self, root, crit=None):
        files = os.listdir(root)
        count = 0
        reqs = []
        for f in files:
            fname = root + '/' + f
            count += 1
            if count % 3000 == 0:
                print "start processing", fname
                print "article num ", count
            data = json.load(open(fname))
            reqs.append(data)
        self.documents = reqs
        if crit is None:
            self.data = [datum['title']+' '+datum['body'] for datum in reqs]
        else:
            assert hasattr(crit, "__call__"), "crit is a parameter that accepts callable only"
            self.data = [crit(datum) for datum in reqs]
        return self.data

    def vectorize_fit(self):
        self.data_mat = self.vectorizer.fit_transform(self.data)

    def fit_model(self):
        self.model.fit(self.data_mat)

    def learn_clusters(self):
        test_data = self.model.transform(self.data_mat)


def main():
    # get articles
    test = articleTest(mSDA(l=1, prob=0.3, multimap=False))
    test.load_data('/Users/PauKung/Dropbox (MIT)/data/data_dumms', crit=lambda d: remove_punc(d['title'].lower()))
    print "start fitting model"
    test.vectorize_fit()
    test.fit_model()
    test.save_model('/Users/PauKung/eventDetect/data/news_model.pkl')
    print "finish fitting model"
    # get development set based on planned parenthood articles
    plist = ['planned parenthood','abortion','defund','fetus']
    # example 1- TF-IDF based clustering

if __name__ == "__main__":
    main()