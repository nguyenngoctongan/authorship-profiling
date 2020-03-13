#!/usr/bin/env python
# coding: utf-8

# In[8]:


import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.utils import shuffle

# In[9]:


def read_xml_files(files, label = "male"):
    """
    Read xml files from the corpus folder to extract 10 raw tweets and match the tweet with its correct label.
    Each xml files content 100 raw tweet from a user
    Input: 
    files: a list of file names
    label: the label of the tweet. Either "male" or "female" according to the truth file in the corpus
    Return: a dataframe including two columns: tweets and labels
    """
    tweets = []
    for file in tqdm(files):
        path = 'Data/pan17/en/'+file+'.xml'
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for child in root.iter('documents'):
            for child2 in child.iter('document'):
                texts.append(child2.text)
        tweets.extend(texts[40:50])
    content = {'tweets': tweets,
          'labels': label}
    df = pd.DataFrame(content)  
    
    return df


# In[10]:


def prepare_data_file():
    """
    Read the truth file to get a list of xml file names and the label of each file. Then read each file to get tweets and labels.
    Merge the data from female sub-corpus and male sub-corpus
    Input: 
    m_files: list of xml file names that have "male" label
    fm_files: list of xml file names that have "female" label
    m_content: a dataframe that inludes all tweets from xml files with "male" label
    fm_content: a dataframe that inludes all tweets from xml files with "female" label
    Return: name of the file that contents tweets and labels
    """
    m_files = []
    fm_files = []
    #Read truth.txt file to match tweet files with correct labels
    with open ('Data/pan17/en/truth.txt', 'r', encoding ='utf-8' ) as t:
        lines = t.readlines() 
        for line in lines:
            new_line = line.replace('\n', '')
            column = new_line.split(':::')
            file_name = column[0]
            gender = column[1]
            if gender == 'male':
                m_files.append(file_name)
            else:
                fm_files.append(file_name)
   
    print('Reading xml files and getting tweets...')
    m_content = read_xml_files(m_files)
    fm_content = read_xml_files(fm_files, label = 'female')

    print("Creating new data file...")
    #Create a new csv file with tweets and labels
    df = pd.concat([m_content, fm_content]) 
    shuffled_df = shuffle(df)
    file_name = 'Data/data.csv'
    shuffled_df.to_csv(file_name, sep = '\t', encoding = 'utf-8')
    print(file_name)
    
    return file_name


# In[11]:


def read_data():
    """
    Get the name of the data file and read the file to get tweets and labels
    file_name: name of the data file that get from running function prepare_data_file
    tweets: list of raw tweets
    labels: list of labels (either "male" or "female")
    Return two lists: a list of tweets and a list of labels
    """
    file_name = prepare_data_file()
    #print("Reading data file")
    read_data = pd.read_csv(file_name,sep = '\t')
    tweets = list(read_data['tweets'])
    labels = list(read_data['labels'])
    print(f'Dataset size: {len(tweets)}')
    
    return tweets, labels


# In[13]:


#tweets, labels = read_data()


# In[ ]:




