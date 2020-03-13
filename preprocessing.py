#!/usr/bin/env python
# coding: utf-8

# In[126]:


import spacy
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


# In[138]:


class Preprocessing:
    
    def __init__(self, tweets):
        self.tweets = tweets
        
    def pre_processing(self):
        nlp = spacy.load('en_core_web_sm')
        print("Preprocessing: Pos tagging, dependency parsing, remove stop words")
        features = []
        documents = []
        for tweet in tqdm(self):
            doc = nlp(tweet.lower())
            sent_features = []
            tokens = []
            for token in doc:
                token_text = str(token.text)
                tokens.append(token_text)
                pos = str(token.pos_)
                dep = str(token.dep_)
                head = str(token.head)
                if not token.is_stop:
                    token_feature = (token_text, pos, dep, head)
                    sent_features.append(token_feature)
        
            documents.append(tokens)
            features.append(sent_features)
        
        return documents, features


# In[139]:


#tweets = ["I love dogs","I don't like you"]


# In[140]:


#processed_tweets = Preprocessing.pre_processing(tweets)
#print(processed_tweets)


# In[ ]:




