
# coding: utf-8

# In[76]:

from POS_Tagger import PerceptronTagger, AveragedPerceptron
reload(POS_Tagger)
#from POS_Tagger import PerceptronTagger


# In[91]:


class Parse(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append([])
            self.rights.append([])
    
    def add_arc(self, head, child):
        self.heads[child] = head
        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)
            
    SHIFT = 0; RIGHT = 1; LEFT = 2
    MOVES = [SHIFT, RIGHT, LEFT]
 
    def transition(move, i, stack, parse):
        global SHIFT, RIGHT, LEFT
        if move == SHIFT:
            stack.append(i)
            return i + 1
        elif move == RIGHT:
            parse.add_arc(stack[-2], stack.pop())
            return i
        elif move == LEFT:
            parse.add_arc(i, stack.pop())
            return i
        raise GrammarError("Unknown move: %d" % move)
        
class Spacy_Parser(object):

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    #AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    def __init__(self, load=False):
        self.tagger = PerceptronTagger()
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        
    def train_tagger(self, sentences_with_tags, num_iters=5):
        self.tagger.train(sentences_with_tags, nr_iter=num_iters)
        
    def parse(self, words):
        sentence = self.tagger.tag(words)
        tags = sentence.pos_tags()
        n = len(words)
        idx = 1
        stack = [0]
        deps = Parse(n)
        while stack or idx < n:
            features = extract_features(words, tags, idx, n, stack, deps)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            next_move = max(valid_moves, key=lambda move: scores[move])
            idx = transition(next_move, idx, stack, parse)
        sentence.set_heads(parse)#still not sure what parse is
        return sentence
 
    def train_one(self, itn, words, gold_tags, gold_heads):
        #spacy blog says using gold tags is not the move
        n = len(words)
        i = 2; stack = [1]; parse = Parse(n)
        tags = self.tagger.tag(" ".join(words)).pos_tags()
        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            guess = max(valid_moves, key=lambda move: scores[move])
            gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
            best = max(gold_moves, key=lambda move: scores[move])
        self.model.update(best, guess, features)
        i = transition(guess, i, stack, parse)
        # Return number correct
        return len([i for i in range(n-1) if parse.heads[i] == gold_heads[i]])
    
    def train(self, sentences, save_loc=None, nr_iter=5, dont_allow=None):
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
                head = sentence.heads() #indeces
                
                train_one(2, words, tags, heads)
            random.shuffle(sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                         open(save_loc, 'wb'), -1)
        return None

    def get_valid_moves(i, n, stack_depth):
        moves = []
        if i < n:
            moves.append(SHIFT)
        if stack_depth <= 2:
            moves.append(RIGHT)
        if stack_depth <= 1:
            moves.append(LEFT)
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
    # Set up the context pieces --- the word, W, and tag, T, of:
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


# In[55]:

import codecs
#### get training set from UD
def load_tagged_sentences(file_name):
    sentences_w_tags = []
    count = 0
    words=[]
    tags=[]
    on_sentence = False
    for line in codecs.open(file_name, 'r', encoding="utf-8"):
    
        vals = line.split('\t')
        if (len(vals) > 1):
            on_sentence = True
            words.append(vals[1])
            tags.append(vals[3])
        elif (on_sentence):
            on_sentence=False
            sentences_w_tags.append((words, tags))
            words=[]
            tags=[]
    
    return sentences_w_tags # [ (["word", "word", "word"], ["tag", "tag", "tag"]), next sentece...]


# In[56]:

parser = Spacy_Parser() 
eng_train = "../../Data/UD_English/en-ud-train.conllu"

tagged_data = load_tagged_sentences(eng_train)



# In[68]:

parser.train_tagger(tagged_data)


# In[92]:

parser.tagger.tag(" ".join(["these", "are", "some", "words", "to", "be", "tagged", ".","\n","a"]))


# In[90]:




# In[ ]:



