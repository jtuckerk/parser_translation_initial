
# coding: utf-8

# In[ ]:




# In[64]:

class Token:
    
    def __init__(self, orig_token="-", pos_tag="-", head="-",head_label='-', conf=0):
        self.orig = orig_token
        self.pos_tag = pos_tag
        if head.isdigit():
            self.head = int(head)
        else:
            self.head = '-'
            
        self.head_label = head_label
        self.conf = conf
        
    def get_testable_attr_list(self):
        return [("POS tag", self.pos_tag), ("Head Index",self.head), ("Dependency Label", self.head_label)]
    
    def get_attr_list(self):
        return [self.orig, self.pos_tag, self.head,self.head_label, self.conf]
    
    def __str__(self):
        return "Token:"+str(self.attr_list)
        #return "[" + self.orig+ ", "+ self.pos + ", "+ self.head+ "]"


# In[65]:

class Sentence(object):
    def __init__(self, tok_list=None):
        if tok_list is None:
            tok_list = []
        self.token_list=tok_list
    
    def add_token(self, tok):
        self.token_list.append(tok)
        
    def get_token_at(self, index):
        return self.token_list[index]
        
    def get_tokens(self):
        return self.token_list
    
    def words(self):
        return [x.orig for x in self.token_list]
    
    def pos_tags(self):
        return [x.pos_tag for x in self.token_list]
    
    def heads(self):
        return [x.head for x in self.token_list]
    
    def head_strings(self):
        return [self.token_list[x.head].orig for x in self.token_list]
    
    def head_labels(self):
        return [x.head_label for x in self.token_list]
    
    def set_heads(self, heads_list):
        #this is weird
        assert len(self.token_list)-1 == len(heads_list)
        for i, head in enumerate(heads_list):
            self.token_list[i+1].head = head
    
    def set_head_labels(self, head_label_list):
        assert len(self.token_list) == len(heads_label_list)
        for i, head_label in enumerate(heads_label_list):
            self.token_list[i].head_label = head_label
    
    def set_pos_tags(self, pos_list):
        assert len(self.token_list) == len(pos_list)
        for i, tag in enumerate(pos_list):
            self.token_list[i].pos_tag = tag
            
    def set_confidences(self, conf_list):
        assert len(self.token_list) == len(conf_list)
        for i, conf in enumerate(conf_list):
            self.token_list[i].conf = conf
    def clear_tags(self):
        for tok in self.token_list:
            tok = Token(orig_token=tok.orig)
            
    def __len__(self):
        return len(self.token_list)


# In[ ]:



