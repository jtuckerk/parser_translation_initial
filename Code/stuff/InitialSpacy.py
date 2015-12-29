
# coding: utf-8

# In[377]:

"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""
from collections import defaultdict
import pickle
import random


class AveragedPerceptron(object):

    '''An averaged perceptron, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    '''

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        
        for feat, value in features.items():
            
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
           
        maxClass = max(self.classes, key=lambda label: (scores[label], label))
        maxScore = scores[maxClass]
        scores[maxClass] = 0
        secondMaxClass = max(self.classes, key=lambda label: (scores[label], label))
        secondMaxScore = scores[secondMaxClass]

        return maxClass, maxScore-secondMaxScore

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None


def train(nr_iter, examples):
    '''Return an averaged perceptron model trained on ``examples`` for
    ``nr_iter`` iterations.
    '''
    model = AveragedPerceptron()
    for i in range(nr_iter):
        random.shuffle(examples)
        for features, class_ in examples:
            scores = model.predict(features)
            guess, score = max(scores.items(), key=lambda i: i[1])
            if guess != class_:
                model.update(class_, guess, features)
    model.average_weights()
    return model


# In[383]:


from __future__ import absolute_import
import os
import random
from collections import defaultdict
import pickle
import logging

from textblob.base import BaseTagger
from textblob.tokenizers import WordTokenizer, SentenceTokenizer
from textblob.exceptions import MissingCorpusError


PICKLE = "trontagger-0.1.0.pickle"


class PerceptronTagger(BaseTagger):

    '''Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    :param load: Load the pickled model upon instantiation.
    '''

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    #AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    def __init__(self, load=False):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        print "oh hello there"
        self.classes = set()
        if load:
            self.load(self.AP_MODEL_LOC)

    def tag(self, corpus, tokenize=True):
        '''Tags a string `corpus`.'''
        # Assume untokenized corpus has \n between sentences and ' ' between words
        s_split = SentenceTokenizer().tokenize if tokenize else lambda t: t.split('\n')
        w_split = WordTokenizer().tokenize if tokenize else lambda s: s.split()
        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)


        print "HEY HEY"
        prev, prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END
            print context
            for i, word in enumerate(words):
                print i, word
                tag = self.tagdict.get(word)
                confidence = 30
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag, confidence = self.model.predict(features)
                tokens.append((word, tag, confidence))
                prev2 = prev
                prev = tag
        return tokens

    def train(self, sentences, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        "Hi train"
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        prev, prev2 = self.START
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for words, tags in sentences:
                
                context = self.START + [self._normalize(w) for w in words]                                                                     + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    confidence = 30
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess, confidence = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                         open(save_loc, 'wb'), -1)
        return None

    def load(self, loc):
        '''Load a pickled model.'''
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise MissingCorpusError(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _normalize(self, word):
        '''Normalization used in pre-processing.
        - All words are lower cased
        - Digits in the range 1800-2100 are represented as !YEAR;
        - Other digits are represented as !DIGITS
        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features

    def _make_tagdict(self, sentences):
        
        '''Make a tag dictionary for single-tag words.'''
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag


def _pc(n, d):
    return (float(n) / d) * 100


# In[237]:

import textblob
textblob.__file__


# In[3]:

trainFile='/Users/tuckerkirven/Desktop/DR-Spring/Data/UD_English/en-ud-train.conllu'
testFile='/Users/tuckerkirven/Desktop/DR-Spring/Data/UD_English/en-ud-test.conllu'


# In[4]:

#### get training set from UD
sentences_w_tags = []
count = 0
words=[]
tags=[]
for line in open(trainFile):

    vals = line.split('\t')
    if (len(vals) > 1):
        words.append(vals[1])
        tags.append(vals[3])
    else:
        sentences_w_tags.append((words, tags))
        words=[]
        tags=[]
len(sentences_w_tags)


# In[386]:

#Train
pt = PerceptronTagger()
#pt = NLTKTagger()
sentences_w_tags[1:4]


# In[ ]:

import time

start = time.time()
pt.train(sentences_w_tags, nr_iter=5)
end = time.time()

print "time taken = " + str(end - start)


# In[2]:

import os
from spacy.en import English, LOCAL_DATA_DIR, DOC
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(parser=False, entity=False, data_dir=data_dir)


# In[17]:

corpus = "\n".join(" ".join(y) for y in [x[0] for x in sentences_w_tags])

doc = nlp(unicode(corpus))


# In[36]:


n = 0
print doc[n].lemma_
print doc[n].pos_
tags = []
for token in doc:
    if not token.pos_ in tags:
        tags.append(token.pos_)
        
Doc[n].data


# In[367]:

#Get Test Corpus - " " seperated words "\n" seperated sentences
# and get list of correct tags
corpus=""
words=[]
test_correct_tags=[]
for line in open(testFile):

    vals = line.split('\t')
    if (len(vals) > 1):
        words.append(vals[1])
        test_correct_tags.append((vals[1],vals[3]))
    else:
        words.append("\n")
print(len(words))
#corpus=words
corpus = " ".join(words)
print corpus[0:20]


# In[387]:

#test_output_tags = pt.tag(corpus, False)
print(len(test_output_tags))
print(len(test_correct_tags))
test_output_tags[0:15]


# In[369]:

import statistics as s
import copy

test_output_tags_c = copy.deepcopy(test_output_tags)
test_correct_tags_c = copy.deepcopy(test_correct_tags)

correct = 0
conf_right=[]
conf_total =0
conf_wrong=[]
poss_total = len(test_output_tags)
print poss_total
unconfident_total = 0
to_remove = []
for i in range(len(test_output_tags_c)):
    if (test_output_tags_c[i][2] <= 7):
        a=3#print (i, test_correct_tags_c[i], test_output_tags_c[i])
        to_remove.append(i)

#for index in reversed(to_remove):
#    del test_correct_tags_c[index]
#    del test_output_tags_c[index]
    
for i in range(len(test_output_tags_c)):
    if (test_correct_tags_c[i][1] == test_output_tags_c[i][1]):
        conf_right.append(test_output_tags_c[i][2])
        correct+=1
    else:
        conf_wrong.append(test_output_tags_c[i][2])
a = np.array(conf_right)
print "average confidence of right = " + str(s.mean(conf_right))
print "average confidence of wrong = " + str(s.mean(conf_wrong))
    
accc = 100.00* (1.0*correct)/poss_total

print  len(to_remove),poss_total
print "accuracy: " + str(accc) + "%"
print "accuracy adjusted " + str(100.0*correct/(poss_total-len(to_remove)))


# In[350]:

ac = [(1,93.449), (2,93.525),(5,93.53) (6,93.70), (8,93.72), (12,93.7599)]
a = [i[0] for i in ac]
b = [i[1] for i in ac]
a
b = 175.265


# In[169]:


get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(a,b)


# In[205]:

acc_id="AC5387d3c4597807d2de889091148d126c"
auth_tok="1639f28d728c5cd85dfcbd57d231c39c"

from twilio.rest import TwilioRestClient
 
# Find these values at https://twilio.com/user/account
account_sid = "AC5387d3c4597807d2de889091148d126c"
auth_token = "1639f28d728c5cd85dfcbd57d231c39c"
client = TwilioRestClient(account_sid, auth_token)
 
message = client.messages.create(to="+15027949011", from_="+1 502-354-4142",
                                     body="done: accuracry = " + str(accc)+ "%")


# In[4]:

import spacy.util
from spacy.en import English

from spacy.syntax.parser import OracleError
from spacy.syntax.util import Config


# In[6]:

nlp = English()


# In[33]:

tokens = nlp.tokenizer.tokens_from_list([u'Me', u'and', u'you, 'u'hate', u'pizza'])
a=nlp.tagger(tokens)
a=nlp.parser(tokens)


# In[51]:

print tokens[1].head


# In[ ]:



