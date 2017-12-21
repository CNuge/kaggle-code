import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re, string
import scipy

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

train.head()

#fillna
train['comment_text'].fillna("unk", inplace=True)

test['comment_text'].fillna("unk", inplace=True)



#get the list of y values we need to predict
to_predict = list(train.columns[2:])


# get the list of tokenizers 
# this is the way to split text into tokens, 
# also split on whitespace in the tokenize functino
symbols = f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])'


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 
	return re_tok.sub(r' \1 ', s).split()

#make sure that works as advertized
tokenize(train['comment_text'][1])

"""
?CountVectorizer
Convert a collection of text documents to a matrix of token counts

This implementation produces a sparse representation of the counts using
scipy.sparse.csr_matrix.
"""


n = train.shape[0]
vec = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=1500000)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


vec = CountVectorizer(ngram_range=(2,4), tokenizer=tokenize, max_features=2000000)

train_sparse = vec.fit_transform(train['comment_text'])
test_sparse = vec.transform(test['comment_text'])

#save the matrix to a file so we can start back here
scipy.sparse.save_npz('sparse_train_punc.npz', train_sparse)
scipy.sparse.save_npz('sparse_test_punc.npz', test_sparse)


# put a kernel stopping at this point to show how the new kaggle functionality
# of using a different kernel's output as your input

# scipy.sparse.load_npz()