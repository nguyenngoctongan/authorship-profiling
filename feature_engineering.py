#!/usr/bin/env python
# coding: utf-8


import spacy
from textblob import TextBlob, Word
from tweet_processing import Tweet_processing
from sklearn.feature_extraction import DictVectorizer
from sklearn import utils
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pickle



def save_data(data, file_name):
    """
    Save data to a file using Pickle
    Return: file name
    """
    with open(file_name,'wb') as f:
        pickle.dump(data, f)
    return file_name


def load_data(file_name):
    """
    Load data from a file using Pickle
    Return: the data in the form before saved
    """
    with open(file_name,'rb') as f:
        X = pickle.load(f)
    return X


def get_word_list(name="positive"):
    """Get a list of sentiment words
    This function has only one key word argument. Key words can be "positive", "negative", "bad", or "modal"
    If key word is not in the list, a warning massage will be shown
    pos_word_list: list of positive words
    neg_word_list: list of negative words
    bad_word_list: list of profane words
    Return a list of positive, negative, profane words, modal verbs or a warning"""
    if name == "positive":
        with open ('word_list/positive-words.txt', 'r') as infile:
            content = infile.read()
            pos_word_list = content.strip().split('\n')
        return pos_word_list
    elif name == "negative":
        with open ('word_list/negative-words.txt', 'r') as infile:
            content = infile.read()
            neg_word_list = content.strip().split('\n')
        return neg_word_list
    elif name =="bad":
        with open ('word_list/bad-words.txt', 'r') as infile:
            content = infile.read()
            bad_word_list = content.strip().split('\n')
        return bad_word_list
    elif name =="modal":
        with open ('word_list/modal.txt', 'r') as infile:
            content = infile.read()
            modal_list = content.strip().split('\n')
        return modal_list
    else:
        return "Word list not found"




def extract_stylo_features(tweets, save_file_name):
    """
    Extracting stylometric features from the corpus
    tweets: a list of tweets from the data file
    save_file_name: name of file to which extracted feature vectors are save
    pos_word_list: a list of positive words
    neg_word_list: a list of negative words
    bad_word_list: a list of profane words
    modal_list: a list of modal verbs
    stylo_features: a list of extracted stylometric features from all tweets
    features: a list of extracted stylometric features from a tweet
    token_list: list of token in a tweet
    num_tkns: the number of tokens in a tweet
    num_snts: the number of sentences in a tweet
    avg_sent_len: the average sentence length
    sum_token_len: the sum of the length of all tokens
    avg_token_len: the average sentence length
    pos_tags: a list of part-of-speech tags of all tokens in a tweet
    nn_count: count the number of nouns in a tweet
    adj_count: count the number of adjective in a tweet
    v_count: count the number of verbs in a tweet
    adv_count: count the number of adverbs in a tweet
    prn_count : count the number of pronouns in a tweet
    polarity_score: the polarity score of a tweet
    subjectivity_score: the subjectivity score of a tweet
    blob_word: a single token in a string format
    dit_count: count the number of tokens that are digits in a tweet
    spc_count: count the number of tokens that are special characters in a tweet
    cap_count: count the number of tokens that contain capital characters in a tweet
    pos_word_count: count the number of tokens that have positve sentiment scores
    neg_word_count: count the number of tokens that have negative sentiment scores
    bad_word_count: count the number of tokens that are profane words
    modal_verb_count: count the number of tokens that are modal verbs
    num_emojis: the number of emojis in a tweet
    num_hashtags: the number of hashtags in a tweet
    num_users: the number of tagged users in a tweet
    ents: a list of named entities in a tweet
    dict_vtrz: call DictVectorizer function
    stylometric: stylometric feature vectors transformed by a DictVectorizer
    save_file: save the feature vector and return the file name
    
    Return: The name of the file that contains the feature vectors
    """
    pos_word_list = get_word_list(name ="pos")
    neg_word_list = get_word_list(name ="neg")
    bad_word_list = get_word_list(name ="bad")
    modal_list = get_word_list(name ="modal")
    
    print("Extracting stylometric features")
    nlp = spacy.load('en_core_web_sm')
    stylo_features = []
    for tweet in tqdm(tweets):
        features = {}
        blob_text = TextBlob(tweet)  
        token_list = blob_text.words
    
    #getting the number of tokens
        num_tkns = len(token_list)
        features['num_tkns'] = num_tkns
        
    #getting the number of sentences
        num_snts = len(blob_text.sentences)
        features['num_snts'] = num_snts
    
    #geting average sentence length
        avg_sent_len = round(num_tkns/num_snts)
        features['ave_snt_len'] = avg_sent_len
    
    #geting average token length
        sum_tkn_len = sum(len(token) for token in token_list)
        avg_token_len = round(sum_tkn_len/num_tkns)
        features['ave_tnk_len'] = avg_token_len

    #counting pos tags
        pos_tags = blob_text.tags   
        nn_count = 0
        adj_count = 0
        v_count = 0
        adv_count = 0
        prn_count = 0
        verb_tags = ("VB","VBD", "VBG", "VBN", "VBP", "VBZ")
        noun_tags = ('NNP', 'NN', 'NNS', "NNPS")
        adj_tags = ('JJ', 'JJR', "JJS")
        adv_tags = ("RB", "RBR", "RBS", "RP")
        prn_tags = ("PRP", "PRP$")
        for token, tag in pos_tags:
            if tag in verb_tags:
                v_count += 1
            elif tag in noun_tags:
                nn_count += 1
            elif tag in adj_tags:
                adj_count += 1
            elif tag in prn_tags:
                prn_count += 1
            elif tag in adv_tags:
                adv_count += 1
            
        features['num_nn'] = nn_count
        features['num_adj'] = adj_count
        features['num_vrb'] = v_count
        features['num_adv'] = adv_count
        features['num_prn'] = prn_count
   
     #getting tweet polarity score
        polarity_score = round(blob_text.sentiment.polarity, 1)
        features['polarity'] = polarity_score
    
    #getting tweet subjectivity score
        subjectivity_score = round(blob_text.sentiment.subjectivity, 1)
        features['subjectivity'] = subjectivity_score
    
    #getting the number of special tokens
        dit_count = 0
        spc_count = 0
        cap_count = 0
        pos_word_count = 0
        neg_word_count = 0
        bad_word_count = 0
        modal_verb_count = 0
        for token in token_list:
            blob_word = Word(token)
        #getting the number of ditgit tokens
            if blob_word.isdigit() == True:
                dit_count += 1
        #getting the number of tokens which are special characters
            if blob_word.isdigit() == False  and blob_word.isalpha() == False:
                spc_count += 1
        #getting the number of tokens containing capital characters
            if blob_word.islower() == False:
                cap_count += 1
        #getting the number of tokens that have positive sentiment
            if blob_word in pos_word_list:
                pos_word_count += 1
        #getting the number of tokens that have negative sentiment
            if blob_word in neg_word_list:
                neg_word_count += 1
        #getting the number of tokens which are profane words
            if blob_word in bad_word_list:
                bad_word_count += 1
        #getting the number of tokens which are modal verbs
            if blob_word in modal_list:
                modal_verb_count += 1
            
        features['num_dit'] = dit_count
        features['num_spc'] = spc_count
        features['num_cap'] = cap_count
        features['num_pos_w'] = pos_word_count
        features['num_neg_w'] = neg_word_count
    
    #getting the number of emojis
        num_emojis = len(Tweet_processing.get_emojis(tweet)) #from tweet_processing.py
        features['num_emo'] = num_emojis
    
    #getting the number of hashtags
        num_hashtags = len(Tweet_processing.get_hashtags(tweet)) #from tweet_processing.py
        features['num_htg'] = num_hashtags
    
    #getting the number of hashtags
        num_users = len(Tweet_processing.get_users(tweet)) #from tweet_processing.py
        features['num_users'] = num_users
    
    #getting number of named entities
        doc = nlp(tweet)
        ents = list(doc.ents)
        features['num_ents'] = len(ents)
    
        stylo_features.append(features)
    
    dict_vtrz = DictVectorizer(sparse=False)
    #transform extracted features into vectors
    stylometric = dict_vtrz.fit_transform(stylo_features)
    
    print(stylometric.shape)
    #save feature vectors
    save_file = save_data(stylometric, save_file_name)
    print(save_file)
    
    return save_file

def tokenised_tweet(tweets):
    """
    Tokenise tweet with Spacy.
    Input: a list of tweets from the data file
    Return: a list of tokenised tweets. Each tweet is a list of tokens
    """
    nlp = spacy.load('en_core_web_sm')
    documents = []
    #tokenising each tweet
    for tweet in tqdm(tweets):
        doc = nlp(tweet.lower())
        tokens = [str(token.text) for token in doc]
        documents.append(tokens)
        
    return documents 
    



def train_wmbs(tweets, size=100):
    """
    Train word embeddings using Word2Vec model from Gensim library
    This function has one positional argument "tweets" and one key word function "size"
    tweets: a list of tweets from the data file
    size: the size of the embedding, default value is 100
    documents: a list of tokenised tweets
    w2v_model: trained word2vec model
    w2v_model_file: path to model file
    
    Return the path to model file
    
    References:
    Documentation for Word2Vec model in Gensim library
    https://radimrehurek.com/gensim/models/word2vec.html
    """
    print(f"Training {size} dimension word embedding model")
    #Training model
    w2v_model = Word2Vec(tokenised_tweet(tweets), size=size, window=5,min_count=1,workers=4,sg=0,hs=1,cbow_mean=1,min_alpha=0.01,iter=20)
    w2v_model_file = "model/word2vec.model"
    w2v_model.save(w2v_model_file)
    
    
    return w2v_model_file




def create_wmbs(model_file,tweets,save_file_name, size = 100):
    """
    Create tweet representation using word embedding
    This function has three positional arguments and one keyword argument
    model_file: the path to the saved word embedding model
    tweets: a list of tweets from the data file
    save_file_name: the name of the file to which feature vectors are saved
    size: the size of the embeddings, default value is 100
    model: the word embedding model loaded from the model file
    word_embs: a list of tweet vectors
    wmb_size: the size of each tweet vectors, calculated by the sum of the sizes of token_vect, pos_vect, dep_vect, and head_vect
    token_vect: the word embedding representing a token in a tweet
    pos_vect: the part-of-speech vector of a token in a tweet
    dep_vect: the dependency vector representing the relation of a token with its head in a tweet
    head_vect: the word embedding representing the head of a token in a tweet
    save_file: save the list of feature vectors and return the name of the save file
    
    Return: name of the save file
    """
    print('Create tweet representation using word embedding')
    model = Word2Vec.load(model_file)
    word_embs = []
    wmb_size = size*2+2
    nlp = spacy.load('en_core_web_sm')
    for tweet in tqdm(tweets):
        tweet_vect = np.zeros(wmb_size)
        doc = nlp(tweet.lower())
        for token in doc:
            if not token.is_stop:
                token_vect = model.wv[str(token.text)]
                pos_vect =  np.array([token.pos])
                dep_vect =  np.array([token.dep])
                head_vect = model.wv[str(token.head)]
                word_vect = np.concatenate((token_vect,pos_vect,dep_vect,head_vect))
                tweet_vect += word_vect
        
        word_embs.append(tweet_vect)
    
    save_file = save_data(word_embs, save_file_name)
    return save_file


def train_dmbs(tweets, size = 200):
    """
    Train document embeddings with tweets in the corpus using Doc2Vec model from Gensim library
    tweets: a list of tweet from the data file
    size: the size of the document embeddings
    tagged_documents: a list of tokenised tweets and tags
    cores: count the number of core processors
    model_dbow: document embedding model trained with tweets and dbow model
    model_dbow_file: the path to the model save file
    
    Return the path to the model save file
    
    Reference: 
    Document for Doc2Vec model in Gensim library
    https://radimrehurek.com/gensim/models/doc2vec.html
    Doc2Vec tutorials:
    https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
    https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    https://rare-technologies.com/doc2vec-tutorial/
    """
    print(f'Training {size} dimension Doc2vec model')
    #Tokenise tweets
    documents = tokenised_tweet(tweets)
    tagged_documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(documents)]
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1,vector_size= size,negative=5,min_count=2,workers=cores,alpha = 0.025,min_alpha=0.025)
    model_dbow.build_vocab(tagged_documents)
    
    for epochs in tqdm(range(30)):
        model_dbow.train(utils.shuffle(tagged_documents), total_examples=model_dbow.corpus_count, epochs=1)
        model_dbow.alpha -= 0.0002
        model_dbow.min_alpha = model_dbow.alpha

    model_dbow_file = "model/d2v_model"
    model_dbow.save(model_dbow_file)
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return model_dbow_file




def create_dmbs(model_file, tweets, save_file_name, size = 200):
    """
    Create tweet representation using document embeddings
    model_file: the path to the saved document embedding model
    tweets: a list of tweets from the data file
    save_file_name: the name of the file to which feature vectors are saved
    size: the size of the document embeddings: default value is 200
    model: the word embedding model loaded from the model file
    doc_embs: a list of document embeddings, each of which represents a tweet
    documents: a list of tokenised tweets
    doc_vect: vector representation of a tweet
    save_file: save the list of feature vectors and return the name of the save file
    
    Return: the name of the file to which feature vectors are saved
    """
    print('Creating document embedding')
    model = Doc2Vec.load(model_file)
    documents = tokenised_tweet(tweets)
    doc_embs = []
    for document in tqdm(documents):
        doc_vect = model.infer_vector(document)
        doc_embs.append(doc_vect)
    
    save_file = save_data(doc_embs, save_file_name)
    return save_file



def combine_feats(stylometric_file,word_embs_file,doc_embs_file,comb = "all", directory = "train"):
    """
    Combine different sets of features together. 
    This function has 3 positional arguments and 2 keyword arguments
    stylometric_file: name of the file to which stylometric feature vectors are saved
    word_embs_file: name of the file to which word embedding feature vectors are saved
    doc_embs_file: name of the file to which document embedding feature vectors are saved
    comb: indicate the combination of features. The combination can be: 
        "S_W": stylometric and word embedding features
        "S_D": stylometric and document embedding features
        "W_D": word embeding and document embedding features
        "all": all three sets of feature
    directory: where to save the file containing feature vectors
    stylometric: a list of stylometric feature vectors
    sty_vect: a stylometric feature vector of a tweet
    word_embs: a list of word embeddings representations
    w_vect: a word embedding vector represention of a tweet
    doc_embs: a list of document embeddings
    d_vect: a document embeding vector of a tweet
    *_save_file_name: the name of the file to which combined feature vectors are saved

    Return the path to the combined feature vector save file
    """


    stylometric = load_data(stylometric_file)
    word_embs = load_data(word_embs_file)
    doc_embs = load_data(doc_embs_file)
    print('Combining features')
    if comb == "S_W":
        print('Stylometric and Word embeddings')
        sty_wmb = []
        for sty_vect, w_vect in tqdm(zip(stylometric,word_embs)):
            sty_wmb_vect = np.concatenate((sty_vect,w_vect))
            sty_wmb.append(sty_wmb_vect)
            
        sw_save_file_name = comb+".pkl"
        return save_data(sty_wmb, sw_save_file_name)
    
    elif comb == "S_D":
        print('Stylometric and Document embeddings')
        sty_dmb = []
        for sty_vect, d_vect in tqdm(zip(stylometric,doc_embs)):
            sty_dmb_vect = np.concatenate((sty_vect,d_vect))
            sty_dmb.append(sty_dmb_vect)
        
        sd_save_file_name = comb+".pkl"
        return save_data(sty_dmb, sd_save_file_name)
    
    elif comb == "W_D":
        print('Word embeddings and Document embeddings')
        wmb_dmb = []
        for w_vect, d_vect in tqdm(zip(word_embs, doc_embs)):
            w_d_vect = np.concatenate((w_vect, d_vect))
            wmb_dmb.append(w_d_vect)
            
        wd_save_file_name = comb+".pkl"
        return save_data(wmb_dmb, wd_save_file_name)
    
    elif comb == "all":
        print('Combining all features')
        all_feat = []
        for sty_vect, w_vect, d_vect, in tqdm(zip(stylometric,word_embs, doc_embs)):
            all_vect = np.concatenate((sty_vect,w_vect,d_vect))
            all_feat.append(all_vect)
        
        all_feat_save_file_name = directory+"/"+comb+".pkl"
        return save_data(all_feat, all_feat_save_file_name)
    
    else:
        return None
        
        
        
        
        
        
