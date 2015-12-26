
# coding: utf-8

# In[717]:

from Parser import *
import os
import statistics as s
from nlp_jtk import Token, Sentence
import codecs
import Parser
reload(Parser)
import nlp_jtk
reload(nlp_jtk)


# In[718]:

class Language(object):
    def __init__(self, name, tagged_file_list=None):
        self.tagger_train_iters = 10
        self.parser_train_iters = 15 #?
        self.name = name
        if tagged_file_list is None:
            tagged_file_list = []
        self.tagged_file_list=tagged_file_list
        
        self.supervised_parser = Spacy_Parser()
        self.unsupervised_parser = Spacy_Parser()
        self.train_data_file = None
        
    def setup(self, train=False, test=False):
        if train:
            print "Training supervised "+self.name+" Tagger."
            self.train_supervised_tagger()
            print "Training supervised "+self.name+" Parser."
            self.train_supervised_parser()
                
        if test:
            print self.name + " test results: "
            self.test_supervised()
        
    def train_supervised_tagger(self):
        self.train_data = self.get_train_data_set()
        print str(len(self.train_data)) + " tagger training sentences"
        print str(self.tagger_train_iters) + " training iterations"
        self.supervised_parser.tagger.train(self.train_data, nr_iter=self.tagger_train_iters)
        
    def train_supervised_parser(self):
        if not self.train_data:
            print "need to train tagger first"
            return
        
        print str(len(self.train_data)) + " parser training sentences"
        print str(self.parser_train_iters) + " training iterations"
        self.supervised_parser.train(self.train_data, nr_iter=self.parser_train_iters)
        
    def test_supervised(self):
        test_correct_data = self.get_test_data_set()
        test_guess = self.supervised_parser.parse(self.convert_sentence_list_to_untagged_corpus(test_correct_data))
        self.get_test_results(test_guess, test_correct_data)
        
    
    def test_unsupervised(self):
        test_correct_data = self.get_test_data_set()
        test_guess = self.unsupervised_parser.parse(self.convert_sentence_list_to_untagged_corpus(test_correct_data))
        self.get_test_results(test_guess, test_correct_data)
        pass
    
    def get_test_results(self, guess_tags, correct_tags):
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
        for feature, correct_count in tag_score_dict.iteritems():
            print feature, "accuracy:", (100.0*correct_count)/total_tags
    

    def get_tagged_sentences(self, file_name):
        sentences_w_tags = []
        count = 0
        words=[]
        tags=[]
        sentence_obj = Sentence()
        sentence_obj.add_token(Token(orig_token='<start>'))
        on_sentence = False
        for line in codecs.open(file_name, 'r', encoding="utf-8"):
        
            vals = line.split('\t')
            if (len(vals) > 1):
                on_sentence = True
                tok = Token()
                tok.orig = vals[1]
                tok.pos_tag = vals[3]
                tok.head = int(vals[6])
                tok.head_label = vals[7]
                sentence_obj.add_token(tok)
            elif (on_sentence):
                on_sentence=False
                sentence_obj.add_token(Token(orig_token='ROOT'))
                sentences_w_tags.append(sentence_obj)
                sentence_obj = Sentence()
                sentence_obj.add_token(Token(orig_token='<start>'))
    
        return sentences_w_tags # [ Sentence_obj, Sentence_obj]
    def get_train_data_set(self):
        print "Tagged data: " + str(len(self.tagged_file_list)) + " files"
        print "Picking Largest"
        large_file = ""
        maxFileSize = 0
        for f in self.tagged_file_list:
            if os.stat(f).st_size > maxFileSize:
                large_file = f
                
                maxFileSize = os.stat(f).st_size
                self.train_data_file = f
        print self.train_data_file
        return self.get_tagged_sentences(large_file)
    
    def get_test_data_set(self):
        test_list = []
        for f in self.tagged_file_list:
            if not f == self.train_data_file:
                test_list += self.get_tagged_sentences(f)
        return test_list
    
    
    def convert_sentence_list_to_untagged_corpus(self, sentence_list):
        import copy
        untagged_list = copy.deepcopy(sentence_list)
        for sent in untagged_list:
            sent.clear_tags()
        return untagged_list
        
        
#use intern function for performance enhancement & pad tokens in the appropriate place
def read_conll(loc):
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        heads = [None]; labels = [None]
        for i, (word, pos, head, label) in enumerate(lines):
            words.append(intern(word))
            #words.append(intern(normalize(word)))
            tags.append(intern(pos))
            heads.append(int(head) + 1 if head != '-1' else len(lines) + 1)
            labels.append(label)
        pad_tokens(words); pad_tokens(tags)
        yield words, tags, heads, labels


def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')


# In[719]:

class Translation(object):
    def __init__(self, src_language, tgt_language, src_file, tgt_file, align_file):
        self.src_language = src_language
        self.src_file = src_file
        
        self.tgt_language = tgt_language
        self.tgt_file = tgt_file
        
        self.align_file = align_file


# In[720]:

en_train_file='../../Data/UD_English/en-ud-train.conllu'
en_test_file='../../Data/UD_English/en-ud-test.conllu'
en_3 = '../../Data/UD_English/en-ud-dev.conllu'
en = Language("English", [en_train_file, en_test_file, en_3])


# In[721]:

#en.setup(train=True, test=True)


# In[722]:

en.train_supervised_tagger()


# In[723]:

en.train_supervised_parser()


# In[724]:

en.test_supervised()


# In[ ]:



