
# coding: utf-8

# ## Spacy Tagger implementation
# getting aroung 94% accuracy for english and spanish trained on UD data sets ~12,000 training sentence for english, ~7,000? sentences for spanish<br>
# need to check if I'm doing something wrong, or just need more training samples. Blog claims 97.something% accuracy

# In[22]:

from spacy_imp.POS_Tagger import PerceptronTagger


# # Helper functions: setup, alignment mapping, test/check...etc

# In[93]:

def convert_corpus_to_sentence_list(corpus):
    sentence_list=[]
    for sentence in corpus.split("\n"):
        sentence_list.append(sentence.split(" "))
    return sentence_list

def convert_sentence_list_no_tags_to_corpus(sentence_list):
    return "\n".join(" ".join(x) for x in sentence_list)
    
#obsolete
def convert_tagged_to_train_format(tagged_sent_list):
    train_list = []
    for sent in tagged_sent_list:
        words=[]
        tags=[]
        for tup in sent:
            words.append(tup[0])
            tags.append(tup[1])
        train_list.append((words,tags))
    return train_list
    


# In[24]:

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


# In[36]:

#args sentences_with_tags = [ (["word", "word", "word"], ["tag", "tag", "tag"]), next sentece...]
def train_tagger(tagger, sentences_with_tags, num_iters=5):
    print str(len(sentences_with_tags)) + " training sentences"
    print str(num_iters) + " training iterations"
    tagger.train(sentences_with_tags, nr_iter=num_iters)


# In[65]:

import codecs
# return arg1 sentences with word/tokens seperated by a " " and sentences seperated by "\n" 
# return arg2 word with tag tuple list
def get_test_corpus(file_name):
    corpus=""
    words=[]
    test_correct_tags=[]
    sentence_tags = []
    sentence_count = 0
    on_sentence = False
    for line in codecs.open(file_name,'r', encoding="utf-8"):

        vals = line.split('\t')
        if (len(vals) > 1):
            on_sentence=True
            words.append(vals[1])
            sentence_tags.append((vals[1],vals[3]))
        elif(on_sentence):
            sentence_count +=1
            on_sentence = False
            words.append("\n")
            test_correct_tags.append(sentence_tags)
            sentence_tags = []


    corpus = " ".join(words)
    print str(sentence_count) + " sentences in test corpus"
    return corpus, test_correct_tags


# In[7]:

#expects corpus in the same form as get test corpus returns as arg1
# returns list ["word", "tag", float_confidence]
def tag_tagger(tagger, corpus, dont_allow=None):
    return tagger.tag(corpus, False, dont_allow)


# In[8]:

import statistics as s
import copy

#todo get accuracy of tags above certain min_confidence_threshold
def analyze_tags(guess_tags, correct_tags, show_full=False, sort_key=lambda ((key_right,key_wrong), value): value):
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
        for word_idx, word_tag_tuple in enumerate(correct_sentence):
            guess_tuple = guess_tags[sent_num][word_idx]
            word = guess_tuple[0]
            tag_guess = guess_tuple[1]
            guess_confidence = guess_tuple[2]
            total_tags +=1
            
            if(word_tag_tuple[1] != tag_guess):
                total_wrong_tags +=1
                conf_wrong.append(guess_confidence)
                perfect_sentence = False
                error_tuple = (word_tag_tuple[1], tag_guess)
                wrong_tag_type[error_tuple] = wrong_tag_type.get(error_tuple, 0) + 1
            else:
                correct_tag_type[tag_guess] = correct_tag_type.get(tag_guess, 0) + 1
                conf_right.append(guess_confidence)
                
        if not perfect_sentence:
            total_wrong_sent+= 1
    
    if(show_full):
        for tag_tup, count in sorted(wrong_tag_type.iteritems(),key=sort_key):
            print "correct:\t"+tag_tup[0]+"\tincorrect:\t"+tag_tup[1]+"\tcount:\t"+str(count)
    print total_wrong_sent, total_sentences
    
    if(len(conf_right) >0 and len(conf_wrong)>0): 
        print "average confidence of right = " + str(s.mean(conf_right))
        print "average confidence of wrong = " + str(s.mean(conf_wrong))
        print "stdev confidence of right = " + str(s.stdev(conf_right))
        print "stdev confidence of wrong = " + str(s.stdev(conf_wrong))
   
    word_acc = (100.00*(total_tags-total_wrong_tags))/total_tags
    sentence_acc = (100.00*(total_sentences-total_wrong_sent))/total_sentences

    
    print "token accuracy: " + str(word_acc) + "%"
    print "sentence accuracy: " + str(sentence_acc) + "%"



# In[85]:

import codecs
# loads src and target original documents and loads alignments into list of tuples
def get_alignment_info(source_file, tgt_file, align_file, num_matches=1000):
    sentence_word_mappings =[]
    orig_sentences = []
    target_sentences= []
    total=0
    matches=0

    from itertools import izip

    with codecs.open(align_file, 'r', encoding="utf-8") as align, codecs.open(source_file, 'r', encoding="utf-8") as orig, codecs.open(tgt_file, 'r', encoding="utf-8") as tgt: 
        for x, y, z in izip(align, orig, tgt):
        
            pairings = []
            for pair in x.split(" "):
                indexs = pair.split("-")
                if(len(indexs) <=1 or (indexs[0] == "" or indexs[1] == "")):
                    continue
                pairings.append((int(indexs[0]), int(indexs[1])))
            src_tokens = y.strip().split(" ")
            tgt_tokens = z.strip().split(" ")
            
            if (not filter_alignments(src_tokens, tgt_tokens, pairings)):
                sentence_word_mappings.append(pairings)
                orig_sentences.append(src_tokens)
                target_sentences.append(tgt_tokens)
                matches+=1
         
          
            if matches>=num_matches:
                break
            total +=1
    print  str((100.0*matches)/total) + "% left after filter. "+ str(matches) + " found after filter"
    print len(orig_sentences)
    print len(target_sentences)
    print len(sentence_word_mappings)
    return orig_sentences, target_sentences, sentence_word_mappings


# In[10]:

#some sort of check to see if the alignment is "good" enough, filters if not
def filter_alignments(src_sent_list, tgt_sent_list, align_pairing_list):
    #dont filter any sentences
    #return False
    
    #filter if length of the target and source are different or if the source and pairings lengths dont match
    #return not (len(src_sent_list) == len(tgt_sent_list) or len(src_sent_list) == len(align_pairing_list))
    
    #filter if there are n fewer pairings than words in the target sentence
    n=1
    return len(tgt_sent_list)-n > len(align_pairing_list)


# In[11]:

untagged_tag_str = "NOTAG"
#create a sentence list for training from tagged source language file and maps using alignments to the target language
def map_tags(tagged_src, untagged_tgt, alignment_list):
    tagged_tgt =[]
    for sentence in untagged_tgt:
        sent_tag_tuple_list = []
        for word in sentence:
            sent_tag_tuple_list.append((word, untagged_tag_str))
        tagged_tgt.append(sent_tag_tuple_list)
            
    count = 0
    for sent_num, pairings in enumerate(alignment_list):
        for pair in pairings:
            src_tag_idx = pair[0]
            tgt_tag_idx = pair[1]

            word = tagged_tgt[sent_num][tgt_tag_idx][0]
            tagged_tgt[sent_num][tgt_tag_idx] = (word, tagged_src[sent_num][src_tag_idx][1])
    
    return tagged_tgt


# In[28]:

#remove sentence that have a low overall confidence per word - or maybe sentences that contain 1 or more very
# unconfident words - then use that corpus to map to target language 
def filter_tagged_corpus(tagged_src_sents, untagged_corresp_sents, alignments, avg_threshold, word_conf_cutoff):
    to_remove = []
    
    for sent_num, sentence in enumerate(tagged_src_sents):
        conf_sum = 0
        removed = False
        for word_tag_conf_tup in sentence:
            conf = word_tag_conf_tup[2]
            conf_sum += conf
            if conf < word_conf_cutoff:
                removed = True
                
        if(len(sentence)==0):
            removed = True
        elif ((1.0*conf_sum)/len(sentence)) < avg_threshold:
            removed = True
        if removed:
            to_remove.append(sent_num)
   
    orig = len(tagged_src_sents)
    left = orig - len(to_remove)
    
    print(len(tagged_src_sents))
    print(len(untagged_corresp_sents))
    print(len(alignments))
    
    print str((100.0*left)/orig) + "% left after filter. " + str(left) + " sentences"
    for idx in reversed(to_remove):
        del tagged_src_sents[idx]
        del untagged_corresp_sents[idx]
        del alignments[idx]


# In[13]:

start1 = "START1"
start2 = "START2"
end1 = "END1"
end2 = "END2"
def generate_pos_trigrams(tagged_sent_list, ignore_tag=""):
    trigram_count_dict = {}
    for sentence in tagged_sent_list:
        tags = [start1, start2] + [i[1] for i in sentence] + [end1, end2]
        for idx in range(len(tags)-2):
            tri = tags[idx:idx+3]

            if (ignore_tag not in tri):
                tri_tup = tuple(tri)
                trigram_count_dict[tri_tup] = trigram_count_dict.get(tri_tup, 0) + 1
    return trigram_count_dict


# In[35]:

def replace_NOTAG_using_trigram(trigram_dict, partially_tagged_sent_list, notag_str="NOTAG"):
    taglist=[]
    for key in trigram_dict:
        for tag in key:
            if not tag in taglist+[start1,start2,end1,end2]:
                taglist.append(tag)

    for sentence in partially_tagged_sent_list:
        tags = [start1, start2] + [i[1] for i in sentence] + [end1, end2]
        indeces_of_notag = []

        for idx in range(len(tags)-2):
            tri = tags[idx:idx+3]

            if (notag_str in tri):
                
                notag_idx = tags.index(notag_str)
                if not notag_idx in indeces_of_notag:
                    indeces_of_notag.append(notag_idx)
        
        for notag_index in indeces_of_notag:
            #tag, tag, notag
            front_tri = tags[notag_index-2:notag_index+1]
            mid_tri = tags[notag_index-1:notag_index+2]
            back_tri = tags[notag_index:notag_index+3]
            
            candidate_tag_score_dict = {}
            
            for tri in [front_tri,mid_tri,back_tri]:
                for potential_tag in taglist:
                    
                    score = trigram_dict.get(tuple([potential_tag if x==notag_str else x for x in tri]),0)
                    candidate_tag_score_dict[potential_tag] = candidate_tag_score_dict.get(potential_tag, 0) + score
                    
            highest_likelyhood_tag = max(candidate_tag_score_dict, key=lambda x: (candidate_tag_score_dict[x],x))

            real_idx = notag_index-2
            word = sentence[real_idx][0]
            sentence[real_idx] = (word, highest_likelyhood_tag)
                                         


# In[66]:

untagged_tag_str = "NOTAG"

#english
en_train_file='../Data/UD_English/en-ud-train.conllu'
en_test_file='../Data/UD_English/en-ud-test.conllu'

#spanish
es_train_file='../Data/UD_Spanish/es-ud-train.conllu'
es_test_file='../Data/UD_Spanish/es-ud-test.conllu'

#arabic...

trainFile=en_train_file
testFile=en_test_file


# ### Load, Train and Test source tagger

# In[67]:

src_language_train_data = load_tagged_sentences(trainFile)


# In[17]:

src_language_tagger = PerceptronTagger()
train_tagger(src_language_tagger, src_language_train_data)


# In[18]:

src_language_init_test_data, src_test_sentence_w_correct_tags = get_test_corpus(testFile)


# In[19]:

src_guess_test_tags = tag_tagger(src_language_tagger, src_language_init_test_data)


# In[80]:

src_guess_test_tagsess_test_tags[:5]


# In[20]:

# results
analyze_tags(src_guess_test_tags, src_test_sentence_w_correct_tags)


# In[99]:

src_text_file = "../Data/UN/c.true.en.en_2_es"
tgt_text_file = "../Data/UN/c.true.es.en_2_es"
align_file = "../Data/UN/aligned.intersect.en_2_es"
num_sents = 75000


# ### Get alignments (do some filtering), tag source language, map to target language

# In[100]:

import time
start = time.time()
src_sent_list, tgt_sent_list, alignments_list = get_alignment_info(src_text_file, tgt_text_file, align_file, num_sents)
end = time.time()
print end-start


# In[101]:

tagged_source = tag_tagger(src_language_tagger, convert_sentence_list_no_tags_to_corpus(src_sent_list))


# In[102]:

src_sent_list[:3]


# In[103]:

filter_tagged_corpus(tagged_source, tgt_sent_list, alignments_list, 20, 0)

untagged_target = tgt_sent_list
tagged_target_data = map_tags(tagged_source, untagged_target, alignments_list)


# In[104]:

pos_trigram_dict = generate_pos_trigrams(tagged_target_data, "NOTAG")

replace_NOTAG_using_trigram(pos_trigram_dict, tagged_target_data, "NOTAG")


# In[105]:

tagged_target_data[0:7]


# # Train target language tagger on alignment tagged data, Test

# In[106]:

target_language_tagger = PerceptronTagger()


# In[107]:

train_tagger(target_language_tagger, convert_tagged_to_train_format(tagged_target_data))


# In[109]:

tgt_language_test_data, tgt_test_sentence_w_correct_tags = get_test_corpus(es_train_file)
tgt_guess_test_tags = tag_tagger(target_language_tagger, tgt_language_test_data)

sort_by_right = lambda ((key_right,key_wrong), value): key_right
sort_by_wrong = lambda ((key_right,key_wrong), value): key_wrong
sort_by_count = lambda ((key_right,key_wrong), value): value
analyze_tags(tgt_guess_test_tags, tgt_test_sentence_w_correct_tags, False, sort_by_count)


# # Notes 
# Earlier tests were messed up - getting expected results now.<br><br>
# 
# 15,000 sentence intermediate - filter sentences with a difference in tokens and alignments > n=1 76.07% accuracy <br>
# 15,000 sentence intermediate - filter alignments if != length or source length != alignment_length 63%  <br>
# 15,000 sentence intermediate - no filter 56% <br><br>
# 30,000 sentence intermediate - filter sentences with a difference in tokens and alignments > n=1 76.82% accuracy <br>
# 
# ### filtering target tagged text used in alignment based on confidence: 77.68
# 2 thresholds: avg_sent = average token confidence in a sentence threshold - filter if below <br>
# min_token = minimum allowed token threshold - filter whole sentence if any token is below<br><br>
# 15000->6513 - n=1 - avg_sent=75% of average correct confidence - min_token=33% of average conf of wrong 74.9% <br> 
# 30,000->6141 - n=1 - avg_sent=99% of average correct confidence - min_token=0 of average conf of wrong 75.93% <br>
# 75,000->15,274 - n=1 - avg_sent=99% of average correct confidence - min_token=0 of average conf of wrong 78.16% <br>
# 30,000->13113 - n=1 - avg_sent=75% of average correct confidence - min_token=33% of average conf of wrong 76.57% <br>
# 36,000->14760 - n=1 - avg_sent=0 of average correct confidence - min_token=100% of average conf of wrong 75.88% <br>

# # English Tagger Generated from English Tagged Data
# ### To see how well this should work with perfect alignments
# #### 83.4% accuracy - down from 93.5% - trained on 15,000 and 12,000 respectively

# In[30]:

en_to_en_tagger = PerceptronTagger()
train_tagger(en_to_en_tagger, convert_tagged_to_train_format(tagged_source))
en_to_en_test_data, en_to_en_test_sentence_w_correct_tags = get_test_corpus(en_test_file)
en_to_en_guess_test_tags = tag_tagger(en_to_en_tagger, en_to_en_test_data)

analyze_tags(en_to_en_guess_test_tags, en_to_en_test_sentence_w_correct_tags, False, sort_by_count)


# In[ ]:

acc_id="AC5387d3c4597807d2de889091148d126c"
auth_tok="1639f28d728c5cd85dfcbd57d231c39c"

from twilio.rest import TwilioRestClient
 
# Find these values at https://twilio.com/user/account
account_sid = "AC5387d3c4597807d2de889091148d126c"
auth_token = "1639f28d728c5cd85dfcbd57d231c39c"
client = TwilioRestClient(account_sid, auth_token)
 
message = client.messages.create(to="+15027949011", from_="+1 502-354-4142",
                                     body="done: accuracry = " + str(accc)+ "%")


# In[17]:

from spacy_imp.POS_Tagger import test1


# In[23]:

PerceptronTagger()


# In[ ]:



