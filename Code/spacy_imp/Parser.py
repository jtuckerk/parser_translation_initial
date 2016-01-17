
# coding: utf-8

# In[154]:

"""A simple implementation of a greedy transition-based parser. Released under BSD license."""
from os import path
import os
import sys
from collections import defaultdict
import random
import time
import pickle

from nlp_jtk import Token, Sentence
import nlp_jtk
reload(nlp_jtk)

SHIFT = 0; RIGHT = 1; LEFT = 2;
MOVES = (SHIFT, RIGHT, LEFT)
START = ['-START-', '-START2-']
END = ['-END-', '-END2-']


class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default


class Parse(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.labels = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))

    def add(self, head, child, label=None):
        self.heads[child] = head
        self.labels[child] = label
        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)


class Spacy_Parser(object):
    def __init__(self, load=False):

        self.model = AveragedPerceptron(MOVES)
        if load:
            self.model.load(path.join(model_dir, 'parser.pickle'))
        self.tagger = PerceptronTagger(load=load)
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def save(self):
       
        self.tagger.save()
    
    def parse(self, corpus):
        sentences = self.tagger.tag(corpus)
        for sentence in sentences:
            words = sentence.words()
            tags = sentence.pos_tags()
            n = len(words)
            i = 2; stack = [1]; parse = Parse(n)
       
            while stack or (i+1) < n:
                features = extract_features(words, tags, i, n, stack, parse)
                scores = self.model.score(features)
                valid_moves = get_valid_moves(i, n, len(stack))
                guess = max(valid_moves, key=lambda move: scores[move])
                i = transition(guess, i, stack, parse)
                
            sentence.set_heads(parse.heads)
        return sentences

    def train_one(self, itn, words, gold_tags, gold_heads):
        n = len(words)
        i = 2; stack = [1]; parse = Parse(n)

        tags, confidences = self.tagger.tag_one(words) # todo may need to update tagger

        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)

            scores = self.model.score(features)

            valid_moves = get_valid_moves(i, n, len(stack))

            gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            assert gold_moves
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, guess, features)
            
            i = transition(guess, i, stack, parse)
            self.confusion_matrix[best][guess] += 1
        return len([i for i in range(n-1) if parse.heads[i] == gold_heads[i]])

    def train(self, sentences, nr_iter, tagger_train_iters=5):

        for itn in range(nr_iter):
            corr = 0; total = 0
            random.shuffle(sentences)
            
            for sent in sentences:
                
                words=sent.words()
                gold_tags = sent.pos_tags()
                gold_parse = sent.heads()
                
                gold_label = sent.head_labels()

                corr += self.train_one(itn, words, gold_tags, gold_parse)

                if itn < tagger_train_iters:
                    self.tagger.train_one(words, gold_tags)
                total += len(words)
           
            print "a", itn, '%.3f' % (float(corr) / float(total))
            if itn == 4:
                self.tagger.model.average_weights()
        print 'Averaging weights'
        self.model.average_weights()



def transition(move, i, stack, parse):
    if move == SHIFT:
        stack.append(i)
        return i + 1
    elif move == RIGHT:
        parse.add(stack[-2], stack.pop())
        return i
    elif move == LEFT:
        parse.add(i, stack.pop())
        return i
    assert move in MOVES


def get_valid_moves(i, n, stack_depth):
    moves = []
    if (i+1) < n:
        moves.append(SHIFT)
    if stack_depth >= 2:
        moves.append(RIGHT)
    if stack_depth >= 1:
        moves.append(LEFT)
    return moves


def get_gold_moves(n0, n, stack, heads, gold):
    def deps_between(target, others, gold):
        for word in others:
            if gold[word] == target or gold[target] == word:
                return True
        return False

    valid = get_valid_moves(n0, n, len(stack))
    if not stack or (SHIFT in valid and gold[n0] == stack[-1]):
        return [SHIFT]
    if gold[stack[-1]] == n0:
        return [LEFT]
    costly = set([m for m in MOVES if m not in valid])
    # If the word behind s0 is its gold head, Left is incorrect
    if len(stack) >= 2 and gold[stack[-1]] == stack[-2]:
        costly.add(LEFT)
    # If there are any dependencies between n0 and the stack,
    # pushing n0 will lose them.
    if SHIFT not in costly and deps_between(n0, stack, gold):
        costly.add(SHIFT)
    # If there are any dependencies between s0 and the buffer, popping
    # s0 will lose them.
    if deps_between(stack[-1], range(n0+1, n-1), gold):
        costly.add(LEFT)
        costly.add(RIGHT) 
        
    moves = [m for m in MOVES if m not in costly]
    if not moves:
        moves = [0]
    return moves


def extract_features(words, tags, n0, n, stack, parse):
    def get_stack_context(depth, stack, data):
        if depth >= 3:
            return data[stack[-1]], data[stack[-2]], data[stack[-3]]
        elif depth >= 2:
            return data[stack[-1]], data[stack[-2]], ''
        elif depth == 1:
            return data[stack[-1]], '', ''
        else:
            return '', '', ''

    def get_buffer_context(i, n, data):
        if i + 1 >= n:
            return data[i], '', ''
        elif i + 2 >= n:
            return data[i], data[i + 1], ''
        else:
            return data[i], data[i + 1], data[i + 2]

    def get_parse_context(word, deps, data):
        if word == -1:
            return 0, '', ''
        deps = deps[word]
        valency = len(deps)
        if not valency:
            return 0, '', ''
        elif valency == 1:
            return 1, data[deps[-1]], ''
        else:
            return valency, data[deps[-1]], data[deps[-2]]

    features = {}
    # Set up the context pieces --- the word (W) and tag (T) of:
    # S0-2: Top three words on the stack
    # N0-2: First three words of the buffer
    # n0b1, n0b2: Two leftmost children of the first word of the buffer
    # s0b1, s0b2: Two leftmost children of the top word of the stack
    # s0f1, s0f2: Two rightmost children of the top word of the stack

    depth = len(stack)
    s0 = stack[-1] if depth else -1

    Ws0, Ws1, Ws2 = get_stack_context(depth, stack, words)
    Ts0, Ts1, Ts2 = get_stack_context(depth, stack, tags)
   
    Wn0, Wn1, Wn2 = get_buffer_context(n0, n, words)
    Tn0, Tn1, Tn2 = get_buffer_context(n0, n, tags)
    
    Vn0b, Wn0b1, Wn0b2 = get_parse_context(n0, parse.lefts, words)
    Vn0b, Tn0b1, Tn0b2 = get_parse_context(n0, parse.lefts, tags)
    
    Vn0f, Wn0f1, Wn0f2 = get_parse_context(n0, parse.rights, words)
    _, Tn0f1, Tn0f2 = get_parse_context(n0, parse.rights, tags)
  
    Vs0b, Ws0b1, Ws0b2 = get_parse_context(s0, parse.lefts, words)
    _, Ts0b1, Ts0b2 = get_parse_context(s0, parse.lefts, tags)

    Vs0f, Ws0f1, Ws0f2 = get_parse_context(s0, parse.rights, words)
    _, Ts0f1, Ts0f2 = get_parse_context(s0, parse.rights, tags)
    
    # Cap numeric features at 5? 
    # String-distance
    Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0

    features['bias'] = 1
    # Add word and tag unigrams
    for w in (Wn0, Wn1, Wn2, Ws0, Ws1, Ws2, Wn0b1, Wn0b2, Ws0b1, Ws0b2, Ws0f1, Ws0f2):
        if w:
            features['w=%s' % w] = 1
    for t in (Tn0, Tn1, Tn2, Ts0, Ts1, Ts2, Tn0b1, Tn0b2, Ts0b1, Ts0b2, Ts0f1, Ts0f2):
        if t:
            features['t=%s' % t] = 1

    # Add word/tag pairs
    for i, (w, t) in enumerate(((Wn0, Tn0), (Wn1, Tn1), (Wn2, Tn2), (Ws0, Ts0))):
        if w or t:
            features['%d w=%s, t=%s' % (i, w, t)] = 1

    # Add some bigrams
    features['s0w=%s,  n0w=%s' % (Ws0, Wn0)] = 1
    features['wn0tn0-ws0 %s/%s %s' % (Wn0, Tn0, Ws0)] = 1
    features['wn0tn0-ts0 %s/%s %s' % (Wn0, Tn0, Ts0)] = 1
    features['ws0ts0-wn0 %s/%s %s' % (Ws0, Ts0, Wn0)] = 1
    features['ws0-ts0 tn0 %s/%s %s' % (Ws0, Ts0, Tn0)] = 1
    features['wt-wt %s/%s %s/%s' % (Ws0, Ts0, Wn0, Tn0)] = 1
    features['tt s0=%s n0=%s' % (Ts0, Tn0)] = 1
    features['tt n0=%s n1=%s' % (Tn0, Tn1)] = 1

    # Add some tag trigrams
    trigrams = ((Tn0, Tn1, Tn2), (Ts0, Tn0, Tn1), (Ts0, Ts1, Tn0), 
                (Ts0, Ts0f1, Tn0), (Ts0, Ts0f1, Tn0), (Ts0, Tn0, Tn0b1),
                (Ts0, Ts0b1, Ts0b2), (Ts0, Ts0f1, Ts0f2), (Tn0, Tn0b1, Tn0b2),
                (Ts0, Ts1, Ts1))
    for i, (t1, t2, t3) in enumerate(trigrams):
        if t1 or t2 or t3:
            features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1

    # Add some valency and distance features
    vw = ((Ws0, Vs0f), (Ws0, Vs0b), (Wn0, Vn0b))
    vt = ((Ts0, Vs0f), (Ts0, Vs0b), (Tn0, Vn0b))
    d = ((Ws0, Ds0n0), (Wn0, Ds0n0), (Ts0, Ds0n0), (Tn0, Ds0n0),
         ('t' + Tn0+Ts0, Ds0n0), ('w' + Wn0+Ws0, Ds0n0))
    for i, (w_t, v_d) in enumerate(vw + vt + d):
        if w_t or v_d:
            features['val/d-%d %s %d' % (i, w_t, v_d)] = 1
    return features


class AveragedPerceptron(object):
    
    '''An averaged perceptron, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    '''

    def __init__(self, classes):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        
        self.weights = {}
        self.classes = classes
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features, dont_allow=None):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        
        for feat, value in features.items():
            
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
        sort_by_score = lambda d: (d[1], d)
        
        first_found=False
        maxClass = "None"
        maxScore = 0
    
        secondMaxClass = "None"
        secondMaxScore = 0
        
        for label, score in sorted(scores.iteritems(), key=sort_by_score, reverse=True):
            if(label != dont_allow and not first_found):
                maxClass = label
                maxScore = score
                first_found=True
            elif(label != dont_allow and first_found):
                secondMaxClass = label
                secondMaxScore = score
                break
      
        return maxClass, maxScore-secondMaxScore
    
    def score(self, features):
        all_weights = self.weights

        scores = dict((clas, 0) for clas in self.classes)
        for feat, value in features.items():
            if value == 0:
                continue
            if feat not in all_weights:
                continue
            weights = all_weights[feat]
            for clas, weight in weights.items():
                scores[clas] += value * weight
        return scores

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


class PerceptronTagger(AveragedPerceptron):

    '''Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    :param load: Load the pickled model upon instantiation.
    '''

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    #AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    def __init__(self, load=False):
        self.classes = set()
        self.model = AveragedPerceptron(self.classes)
        self.tagdict = {}
        
        if load:
            self.load(self.AP_MODEL_LOC)

    def tag(self, corpus, tokenize=False, dont_allow=None):
        '''Tags a string `corpus`.'''
        # Assume untokenized corpus has \n between sentences and ' ' between words
        prev, prev2= self.START
        sentences = []
        for sent in corpus:
            tagged_sentence = Sentence()
          
            words = sent.words()
            context = self.START + [self._normalize(w) for w in words] + self.END
            for i, word in enumerate(sent.get_tokens()):
                tag = self.tagdict.get(word)
                confidence = 30
                if not tag:
                    features = self._get_features(i, word.orig, context, prev, prev2)
                    tag, confidence = self.model.predict(features, dont_allow)
                    
                
                word.pos_tag = tag
                word.conf = confidence
                
                prev2 = prev
                prev = tag
            sentences.append(sent)
        return sentences

    def tag_one(self, words, tokenize=False):
        prev, prev2 = START
        tags = DefaultList('') 
        confidences = []
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            confidence = 25
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, confidence = self.model.predict(features)
            tags.append(tag)
            confidences.append(confidence)
            prev2 = prev; prev = tag
        return tags, confidences

    def tag(self, corpus):
        for sent in corpus:
            tags, confidences = self.tag_one(sent.words())
            sent.set_pos_tags(tags)
            sent.set_confidences(confidences)
        return corpus
    
    def train(self, sentences, save_loc=None, nr_iter=5, dont_allow=None):
        print "THIS TRAIN2"
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        prev, prev2 = self.START
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for sentence in sentences:
                
                words = sentence.words()
                tags = sentence.pos_tags()
                context = self.START + [self._normalize(w) for w in words]                                                                     + self.END
                for i, word in enumerate(words):
                    guess = None # self.tagdict.get(word)
                    confidence = 30
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess, confidence = self.model.predict(feats, dont_allow)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(sentences)

        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                         open(save_loc, 'wb'), -1)
        return None

    def train(self, sentences, save_loc=None, nr_iter=5):
        print "THIS TRAIN1"
        '''Train a model from sentences, and save it at save_loc. nr_iter
        controls the number of Perceptron training iterations.'''
        self.start_training(sentences)
        for iter_ in range(nr_iter):
            for sent in sentences:
                words = sent.words()
                tags = sent.pos_tags()
                self.train_one(words, tags)
            random.shuffle(sentences)
        
    def save(self):
        # Pickle as a binary file
        pickle.dump((self.model.weights, self.tagdict, self.classes),
                    open(PerceptronTagger.model_loc, 'wb'), -1)

    def train_one(self, words, tags):
        prev, prev2 = START
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            guess = self.tagdict.get(word)
            if not guess:
                feats = self._get_features(i, word, context, prev, prev2)
                guess, confidence = self.model.predict(feats)
                self.model.update(tags[i], guess, feats)
            prev2 = prev; prev = guess

    def load(self, loc):
        print '''Load a pickled model.'''
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
            return word #.lower()

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

        for sentence in sentences:
            words = sentence.words()
            tags = sentence.pos_tags()
            
            for word, tag in zip(words, tags):
                counts[word][tag] += 1

                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.93
        print "AMBIGUITY THRESH: ", ambiguity_thresh
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag


def _pc(n, d):
    return (float(n) / d) * 100

def read_pos(loc):
    for line in open(loc):
        if not line.strip():
            continue
        words = DefaultList('')
        tags = DefaultList('')
        for token in line.split():
            if not token:
                continue
            word, tag = token.rsplit('/', 1)
            #words.append(normalize(word))
            words.append(word)
            tags.append(tag)
        pad_tokens(words); pad_tokens(tags)
        yield words, tags


#use intern function for performance enhancement & pad tokens in the appropriate place
def read_conll(loc):
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        heads = [None]; labels = [None]
        for index, word,lem, pos, something, s1, head, label, s2, s3 in lines:
            words.append(intern(word))
            #words.append(intern(normalize(word)))
            tags.append(intern(pos))
            heads.append(int(head) if head != '0' else len(lines) )
            labels.append(label)
        pad_tokens(words); pad_tokens(tags)

        sent_obj = Sentence()
        sent_obj.add_words(words)
        sent_obj.set_pos_tags(tags)
        sent_obj.set_heads(heads)
        sent_obj.set_head_labels(labels)
        
        yield sent_obj


def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')

    


# In[161]:

train_set = "../../Data/test_pars_data/en-ud-train.conllu"
test_set = "../../Data/test_pars_data/en-ud-test.conllu"
heldout_in = "../../Data/test_pars_data/held_in.txt"


# In[162]:

input_sents = list(read_pos(heldout_in))
parser = Spacy_Parser(load=False)
sentences = list(read_conll(train_set))
 
parser.train(sentences, nr_iter=10)


# In[163]:

gold_sents = list(read_conll(test_set))
    
import copy
blank = copy.deepcopy(gold_sents)
for sentence in blank:
    sentence.clear_tags()

heads = parser.parse(blank)

import statistics as s
def get_test_results(guess_tags, correct_tags):
    tag_score_dict = {}
        
    correct_tag_type ={}
    wrong_tag_type = {}
    
    conf_right = []
    conf_wrong = []
    
    total_tags = 0
    total_wrong_tags = 0
    
    total_sentences = len(guess_tags)
    total_wrong_sent = 0
    
    for sent_num, correct_sentence in enumerate(correct_tags):

        perfect_sentence = True
        for word_idx, correct_token in enumerate(correct_sentence.get_tokens()):
            guess_token = guess_tags[sent_num].get_token_at(word_idx)
            assert correct_token.orig == guess_token.orig
                
            for i, (feature, guess) in enumerate(guess_token.get_testable_attr_list()):
                tag_score_dict[feature] = tag_score_dict.get(feature, 0) + (guess==correct_token.get_testable_attr_list()[i][1])
                
            tag_guess = guess_token.pos_tag
            guess_confidence = guess_token.conf
            total_tags +=1
            
            if(correct_token.pos_tag != tag_guess):
                total_wrong_tags +=1
                conf_wrong.append(guess_confidence)
                perfect_sentence = False
                error_tuple = (correct_token.pos_tag, tag_guess)
                wrong_tag_type[error_tuple] = wrong_tag_type.get(error_tuple, 0) + 1
            else:
                correct_tag_type[tag_guess] = correct_tag_type.get(tag_guess, 0) + 1
                conf_right.append(guess_confidence)
                
        if not perfect_sentence:
            total_wrong_sent+= 1
                
    if(len(conf_right) >0 and len(conf_wrong)>0): 
        print "average confidence of right tag= " + str(s.mean(conf_right))
        print "average confidence of wrong tag= " + str(s.mean(conf_wrong))
        print "stdev confidence of right tag= " + str(s.stdev(conf_right))
        print "stdev confidence of wrong tag= " + str(s.stdev(conf_wrong))
   
    tag_word_acc = (100.00*(total_tags-total_wrong_tags))/total_tags
    tag_sentence_acc = (100.00*(total_sentences-total_wrong_sent))/total_sentences

    print "tag token accuracy: " + str(tag_word_acc) + "%"
    print "tag sentence accuracy: " + str(tag_sentence_acc) + "%"
    print "have not written tests for parse yet"
    for attribute, correct_count in tag_score_dict.iteritems():
        print attribute, "accuracy:", (100.0*correct_count)/total_tags

get_test_results(heads, gold_sents)


# In[ ]:



