
# coding: utf-8

# In[ ]:




# In[23]:

class Token:
    
    def __init__(self, orig_token="-", pos_tag="-", head="-", conf=0):
        self.orig = orig_token
        self.pos_tag = pos_tag
        if head.isdigit():
            self.head = int(head)
        else:
            self.head = '-'
        self.conf = conf
        self.attr_list = [self.orig, self.pos_tag, self.head, self.conf]
        self.testable_attr = [self.orig, self.pos_tag, self.head]
        
    def __str__(self):
        return "Token:"+str(self.attr_list)
        #return "[" + self.orig+ ", "+ self.pos + ", "+ self.head+ "]"


# In[24]:

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
    
    def set_heads(heads_list):
        assert len(self.token_list) == len(heads_list)
        for i, head in enumerate(heads_list):
            self.token_list[i].head = head
    
    def set_pos_tags(pos_list):
        assert len(self.token_list) == len(pos_list)
        for i, head in enumerate(heads_list):
            self.token_list[i].pos_tag = head
            
    def __len__(self):
        return len(self.token_list)


# In[ ]:



