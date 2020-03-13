#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from reading_data_files import read_data
from feature_engineering import train_wmbs,train_dmbs,extract_stylo_features,create_wmbs,create_dmbs,combine_feats
from classifier import classify
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#Read data from xml files
tweets, labels = read_data() #from read_data_files.py

#Training embeddings models
#Word embedding model
wmb_model = train_wmbs(tweets)#from feature_engineering.py
#Document embedding model
dmb_model = train_dmbs(tweets)#from feature_engineering.py


print('Spliting data...')
X_train, X_test, y_train, y_test = train_test_split(tweets,labels,test_size=0.25)
print(f"Train size: {len(X_train)}\nTest size:{len(X_test)}\n")

train_df = pd.DataFrame({"tweet": X_train,"label":y_train})
test_df = pd.DataFrame({"tweet":X_test, "label": y_test})

train_file = train_df.to_csv("Data/train.csv", sep = "\t", encoding = "utf-8")
test_file = test_df.to_csv("Data/test.csv", sep = "\t", encoding = "utf-8")

print("Extracting and representing features")
#Stylometric features
train_stylo_vectors = extract_stylo_features(X_train,"train/train_stylo.pkl")#from feature_engineering.py
test_stylo_vectors = extract_stylo_features(X_test,"test/test_stylo.pkl")#from feature_engineering.py

#Word embedding features
train_word_embs = create_wmbs(wmb_model, X_train,"train/train_wmbs.pkl")#from feature_engineering.py
test_word_embs = create_wmbs( wmb_model, X_test,"test/test_wmbs.pkl")#from feature_engineering.py

#Document embedding features
train_doc_embs = create_dmbs(dmb_model, X_train,"train/train_doc.pkl")#from feature_engineering.py
test_doc_embs = create_dmbs(dmb_model, X_test,"test/test_doc.pkl")#from feature_engineering.py

#Combined features
all_combined_train = combine_feats(train_stylo_vectors,train_word_embs,train_doc_embs, comb = "all", directory = "train")#from feature_engineering.py
all_combined_test = combine_feats(test_stylo_vectors,test_word_embs,test_doc_embs, comb = "all", directory = "test")#from feature_engineering.py

print("\nTraining algorithms using stylometric features")
stylo_result = classify(train_stylo_vectors, y_train,test_stylo_vectors, y_test,feature = "stylometric")#from classifier.py

print("\nTraining algorithms using word embedding features")
wmb_result = classify(train_word_embs, y_train,test_word_embs, y_test,feature = "wmb")#from classifier.py

print("\nTraining algorithms using document embedding features")
dmb_result = classify(train_doc_embs, y_train, test_doc_embs, y_test,feature = "dmb")#from classifier.py

print("\nTraining algorithms using all features")
all_result = classify(train_doc_embs, y_train, test_doc_embs, y_test, feature = "all")#from classifier.py


