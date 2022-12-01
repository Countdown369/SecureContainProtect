#!/usr/bin/env python
# coding: utf-8

# # Data and Analysis Plan: Secure. Contain. Protect.
# 
# - Lee Fenuccio
# - Rudra Sett
# - Connor Brown

# ## Project Goal:
# This project uses data scraped from the SCP wiki to analyze similarities and differences between SCPs. Using these analyzes, we can build classifiers to predict attributes of unknown SCPs, and recommend different SCPs based on someone's interests. 

# ## Data 
# 
# ### Overview 
# We will scrape data off of the [SCP Wiki](https://scp-wiki.wikidot.com/) to obtain data for each SCP. The SCP Wiki has a unique page for each SCP object that contains information about it. Our pipeline will extract specific attributes of an SCP and put them into a dataframe to be used for analysis. Below is a sample of a page for an SCP off of the SCP wiki, [SCP-4141](https://scp-wiki.wikidot.com/scp-4141). 
# ![Wiki](Wiki_Screenshot.jpg)
# 
# Some easily visible attributes of the SCP are its number (4141), Class (Safe) and Rating (+102). Based on a more in depth knowledge about SCPs, there are other tropes that we searched the scraped text for, such as if the object had any sub-objects or mentions of D-Class personnel. Then we will use some natural language processing to gather more information about the style of how the SCP was written, such as sentiment analysis. 
# 
# ### Pipeline Overview
# 
# We will accomplish this task with the following functions:
# 
# #### Webscraping
# - 'rand_scp_num()'
#     - gets a random number in the form of the scp wiki url
# - 'get_scp_soup()'
#     - gets soup object for a given scp number
# - 'get_scp_maintext()'
#     - gets the main text from the scraped scp
# - 'get_class()'
#     - gets the class of the scraped scp
# - 'get_rating()'
#     - gets the positive and overall ratings from the scraped scp
# - 'get_subobjects()'
#     - get a list of subobjects 1-9 and A-Z that exist in scraped scp
# - 'get_tags()'
#     - gets the tags for the scraped scp
#     
# #### Formatting Data
# - 'get_dict()'
#     - gets the dictionary with attributes for a certain scp
# - 'get_random_dataframe()'
#     - creates a dataframe from a specified number of random scps
#     
# #### Natural Language Processing
# - 'fix_pos()'
#     - corrects the part of speech to something the lemmatizer can actually read
# - 'clean_text()'
#     - cleans a given SCP story for further analysis
# - 'make_wordcloud()'
#     - makes a wordcloud
# - 'get_setiment()'
#     - gets sediment

# ### Pipeline
# #### Webscraping

# In[1]:


# random scp generator
import random

def rand_scp_num(x = 1):
    ''' generates x number of random numbers from 1-5999 that corresponds to an scp page
    Args:
        x = 1 (int): integer for amount of scp numbers you want returned
    Returns:
        scp_num (list): a list with values from 1-5999, formatted to correctly open an scp page
    '''
    # create list to store scp numvers in
    scp_nums = []
    for i in range(x):
        n = "000" + str(random.randint(1, 5999))
        while len(n) > 4:
            n = n.replace("0", "", 1)
        if n[0] == "0":
            n = n.replace("0", "", 1)
        scp_nums.append(n)
    return(scp_nums)


# In[2]:


# function to input scp number into and return soup object for that scp's webpage
import requests
import json
from bs4 import BeautifulSoup

def get_scp_soup(scp):
    '''gets Beautiful Soup of the given scp number
    Args:
        scp (int): the number of the scp 
    Returns:
        scp_soup (Beautiful Soup object): soup object from the scp html
    '''
    # first, it gets the html for the given scp
    url = f'https://scp-wiki.wikidot.com/scp-{scp}'
    html = requests.get(url).text
    
    # get the soup for the html
    scp_soup = BeautifulSoup(html)
    
    # return the soup
    return(scp_soup)


# In[3]:


def get_scp_maintext(num, scp_soup):
    '''gets the main story text from the scp soup object
    Args:
        num (int): the number of the scp the soup is for
        scp_soup (Beautiful Soup object): soup object for the scp's webpage
    Returns:
        scp_story (str): string that contains the story from the scp webpage
    '''
    # get the text using div and page-content from the soup
    scp_fulltext = (scp_soup.find("div", id = "page-content")).text
    
    # the text potentially has things before and after the actual story, so subset this text into just the story
    # we are assuming the story starts once the item is named (Item #: SCP-XXXX)
    # we are assuming the story ends when the bottom of the screen displays links to the next/previous story (« SCP-XXXX - 1)
    
    # create the words that indicate the story starts/stops and find its location in the full text
    start = scp_fulltext.find(f'Item #: SCP-{num}')
    stop = scp_fulltext.find(f'« SCP-{int(num)-1}')
    
    # subset the full text by these location
    scp_story = scp_fulltext[start : stop]
    
    return(scp_story)
    


# In[4]:


# gets positive and overall rating for the scp
def get_rating(soup):
    '''gets the positive and overall ratings for the scp
    Args:
        soup (Beautiful Soup object): soup for the scp
    Return:
        ratings (list): list with positive rating, overall rating    
    '''
    # get the 8th class=image, which is the image of the rating box, and get the src, which is the link
    text = soup.find_all(class_ = "image")[7]["src"]
    
    # clean the link to get the ratings from it
    link = text.split("&")
    ratings = [int(link[3].replace("rating=", "")), int(link[4].replace("rating_votes=", ""))]
    
    return(ratings)


# In[5]:


import string
def get_subobjects(num, text):
    '''gets a list of the letters of all subobjects for an scp
    Args:
        num (int): the number of the scp
        text (str): the text of the main scp story
    Return:
        sub_objects (list): a list of the letters for all sub objects    
    '''
    # create list for subobjects
    sub_objects = []
   
    lets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for j in list(string.ascii_uppercase):
        lets.append(j)
    
    for i in lets:
        is_sub = f'SCP-{num}-{i} ' in text
        # if it exsits, add the letter into list of subobjects and then make the letter the next letter
        if is_sub == True:
            sub_objects.append(i)
            
    if len(sub_objects) > 0:
        return sub_objects
    else:
        return None
    
# to improve subobjects:
#    - find way to get sub-objecets >9, some objects have them


# In[6]:


def get_tags(soup):
    ''' gets the tags on the bottom of the page for the scp
    Args:
        soup (Beautiful Soup object): the beautiful soup object for the scp
    Returns:
        tags (list): a list of the tags
    '''
    # get the part of the soup that has the page tags
    scp_tags = soup.find(class_ = "page-tags")
    
    # the page tags is an amalgamation of things, seperate them by their a to get a list of each tag
    scp_tags_list = scp_tags.find_all("a")

    # loop over the tags and add the text of all the tags to a list
    tags = []
    for tag in scp_tags_list:
        # some tags start with"_", we dont want those, so only add ones that don't start with "_"
        if (tag.text)[0] != "_":
            tags.append(tag.text)
        
    return(tags)


# In[7]:


def get_class(text):
    '''returns class of SCP based on SCP text
    Args:
        text (str): text of SCP page (according to scp_dataframe())
    Return:
        class (str): string of SCP class (e.g. 'Euclid')
    '''
    try:
        start = text.index("Object Class:")
    except ValueError:
        return None
    text = text[start+14:]
    end = text.index("\n")
    
    return text[:end]


# #### Formatting Data

# In[8]:


def get_dict(num):
    ''' get a dictionary with information about different attributes for an scp
    Args:
        num (int): number of scp

    Returns: 
       dic_scp (dict): dictionary with information about different attributes for the scp    
    '''
     # create dic to add scp attributes to
    dic_scp = {}
    
    # get the random scp's soup (Beautiful Soup object)
    soup = get_scp_soup(num)
           
    # get rating of scp
    ratings = get_rating(soup)
        
    # get the tags of the scp
    tags = get_tags(soup)

    # get the main text of the scp
    text = get_scp_maintext(num, soup)
            
    scp_class = get_class(text)
        
    word_count = len(text.split())
        
    # get list of sub objects
    sub_objects = get_subobjects(num, text)
        
    # get boolean of if D-Class is mentioned in text
    dclass = "D-Class" in text
        
        
    # create dic to add scp attributes to, to eventually add to dataframe
    dic_scp = {"Number" : num, "Class" : scp_class, "Pos Ratings" : ratings[0], "Pos Rating Rate" : ratings[0]/ratings[1], 
                   "D-Class" : dclass, "Tags" : tags, "Text" : text, "Sub-Objects" : sub_objects, 
               "Word Count" : word_count, "Sentiment" : None}
    return(dic_scp)


# In[9]:


import pandas as pd

def get_random_dataframe(x=1):
    '''creates dataframe with attributes for a number of random scps
    Args:
        x=1 (int): optional number of how many scps to put in the dataframe
    Return:
        df_scp (df): dataframe with attributes for a number of random scps
    '''  
    
    scp_nums = rand_scp_num(x)
    # create x random scp numbers
    
    # create dataframe with column names but no data
    df_scp = pd.DataFrame(columns = ["Number", "Class", "Pos Ratings", "Pos Rating Rate", "Sub-Objects", 
                                     "D-Class", "Tags", "Text", "Word Count", "Sentiment"])
    
    # loop through each random scp created, and get the dictionary of its attributes
    for num in scp_nums:
        scp_dict = get_dict(num)
        
        # add dict as another row in the scp dataframe, but only if there is text
        if scp_dict["Text"] == '':
            print(f'Access to SCP-{num} is restricted')
        else:
            df_scp = df_scp.append(scp_dict, ignore_index=True)
        
    # set the index as the SCP number in the dataframe    
    df_scp = df_scp.set_index("Number")
    
    return(df_scp)


# #### Natural Language Processing 

# In[13]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import string
import wordcloud
from plotly import express as px


# In[14]:


def fix_pos(pos):
  '''corrects the part of speech to something the lemmatizer can actually read
    Args:
        pos (str): the part of speech returned by the POS tagger
    Return:
        str: the correct type of part of speech
  '''
  # if tag not in ["DT","JJ","POS","WP","PRP$","IN","CC","CD","WRB","PRP","TO","MD",".","WDT"]
  correct = ["A","V","N","R"]
  pos = pos[0]
  if pos in correct:
    return pos.lower()
  else:
    return 'n'


# In[15]:


def clean_text(text):
  '''cleans a given SCP story for further analysis
    Args:
        text (str): the story in the form of a str
    Return:
        cleaned_text (str): the story with special characters removed and 
  '''   
 
  # expand contractions
  text =  contractions.fix(text)

  # make a string translation table (not sure how common this is, but I've used it before; there are definitely lots of other ways to do this!)
  table = str.maketrans(string.punctuation,' '*len(string.punctuation))
  # the reason this is here is because I actually don't want dashes removed
  table.update({ord("-"): '-'})
  text = str.translate(text,table)

  # split the text into words
  tokens = word_tokenize(text)
  # tag the words with part of speech
  tagged_text = pos_tag(tokens)
  # remove very common grammatical words like and, but, for, etc.
  stop_words = stopwords.words("english")
  # convert words to their roots (i.e. staffing to staff)
  lemmatizer = WordNetLemmatizer()
  lemmatized = [lemmatizer.lemmatize(word,fix_pos(tag)) for word, tag in tagged_text if word not in stop_words]

  # join the words back into one document again and return them
  cleaned_text = " ".join(lemmatized)
  return cleaned_text


# In[16]:


def make_wordcloud(doc):
  '''makes a word cloud given a document
    Args:
        doc (str): a single document (one string)
    Return:
        cloud (WordCloud): a WordCloud object
  '''
  cloud = wordcloud.WordCloud()
  cloud.generate_from_text(doc)
  return cloud


# In[17]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
# nltk needs to download a thing to make the sentiment analysis work
nltk.downloader.download('vader_lexicon')


# In[18]:


def get_sentiment(doc):
  '''returns the overall sentiment of a document
    Args:
        doc (str): the document to analyze
    Return:
        float: the sentiment
  '''
  sent = SentimentIntensityAnalyzer()
  return sent.polarity_scores(doc)["compound"]


# #### Running Code

# In[19]:


# make a dataframe with 100 random scps
df_scp = get_random_dataframe(1)
df_scp


# In[20]:


# clean the text and get sentiment for each scp
df_scp["Text"] = df_scp["Text"].apply(clean_text)
df_scp["Sentiment"] = df_scp["Text"].apply(get_sentiment)
df_scp


# In[21]:


# create a word cloud of ALL SCP documents... apparently SCP is the most common word; who could've guessed?
cloud = make_wordcloud(" ".join(df_scp["Text"]))
cloud.to_image()


# In[22]:


# make a bar plot of occurances of different classes
px.bar(df_scp,"Class")


# In[23]:


# make a histogram for sentiment
px.histogram(df_scp,"Sentiment")


# ## Analysis Plan
# We are looking to analyze the data we get from the SCPs and use it to make decisions about what attributes an unknown SCP would have, and an algorithmn to recommend SCPs to read based off of someone's preferences. 
# 
# One algorthimn could be to use a Random Forest to look through an unknown SCP and decide what its class should be. Since the entries all have their class directly written in their text, this will have to be removed from the attribute output so it isn't already known about an SCP. This will involve creating a dataframe of many SCPs and their attributes, and converting all attributes to numerical values. This will be easy for things such as True/False, but more difficult for things such as page tags, since there are so many. Then this data will be implemented into a Random Forest classifier. We can use cross-validation to check its accuracy.
# 
# Another data analysis we are interested in doing but haven't learned in class yet is a recommendation algorithmn. This could be done very similarly to the K-Nearest Neighbor classifier, but instead of using the nearest neighbors to estimate a category, we could save what the n nearest neighbors are and use them as recommendations. Or we could use cosine similarity to assign vectors to SCPs and compare their similarity. This would be similar to Netflix's 'More like this' recommmendations. We give the algorithmn an SCP or a few, and it returns several SCPs that have similar attributes. 
