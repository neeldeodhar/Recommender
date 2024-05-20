#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing pandas
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#reading dataset into a Pandas dataframe

df = pd.read_csv("books.csv", on_bad_lines = "skip")


# In[3]:


#describing the dataset
df.describe()


# In[4]:


df.head(2).transpose()


# In[5]:


#finding highest rated books
top5Recommendations = df.sort_values(by = 'average_rating',
                                     ascending = False).head(5)
top5Recommendations


# In[6]:


#finding highest weighted ranked (by ratings count)books in dataset
top5Votes = df.sort_values(by = 'ratings_count',ascending = False).head(5)
top5Votes


# In[7]:


#finding top 5 books based on text reviews
top5textreviews = df.sort_values(by = 'text_reviews_count',ascending = False).head(5)
top5textreviews


# In[8]:


#listing columns
df.columns


# In[9]:


#creating a function named Popularity Recommender
def popularityRecommender(df):
    
    #Define the minimum ratings count
    minimum_ratings_count = 0.85* df['ratings_count'].max()
    
    #Define the minimum text reviews count
    minimum_text_count = 0.85* df['text_reviews_count'].max()
    
    
    #the mean rating
    mean_rating = df['average_rating'].mean()

    #both 'ratings_count' and 'text_reviews_count' has been used towards calculating 'weighted_rating'
    df['weighted_rating'] = (((df['ratings_count']/(df['ratings_count']+minimum_ratings_count)) * df['average_rating']) +
                             ((minimum_ratings_count/(df['ratings_count']+minimum_ratings_count))*mean_rating) + (df['text_reviews_count']/(df['text_reviews_count']+minimum_text_count)) * df['average_rating']) + ((minimum_ratings_count/(df['text_reviews_count']+minimum_text_count))*mean_rating)

    recommendations = df.sort_values(by = 'weighted_rating',ascending = False).head(5)
    
    return(recommendations) 


# In[10]:


#getting top 5 recommended books
top5 = popularityRecommender(df)
top5[["title",'ratings_count','text_reviews_count','average_rating','weighted_rating',]].head(5)


# In[11]:


#CONTENT BASED RECOMMENDER

from sklearn.feature_extraction.text import TfidfVectorizer
cbr = TfidfVectorizer(stop_words = 'english')

# Replace empty descriptions with a blank "" value and transform the titles of books in our dataset into the matrix
df['title'] = df['title'].fillna('')
tfidf_matrix = cbr.fit_transform(df['title'])

tfidf_matrix.shape


# In[12]:


df['title'][0]


# In[13]:


# Assign the instance of our recommender function.
# This is a matrix with a similarity value for every book with every other book in the dataset

from sklearn.metrics.pairwise import cosine_similarity
distance_matrix = cosine_similarity(tfidf_matrix)

# Re-create the indices of our list of books by removing any duplicates if required
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# In[14]:


# Define a function that takes the re-indexed dataset, finds the 6 most similar titles 
#to a chosen title based on the
# similarity of the words in the titles,
# and returns the top 5, 

def ContentBasedRecommender(title, indices, distance_matrix):
    id_ = indices[title] #Fetch the index of the books we will enter
    
    #List of tuples with distance for each book to the entered book
    distances = list(enumerate(distance_matrix[id_])) 
    
    #sort by the distance function, which is in column[1]
    distances = sorted(distances, key=lambda x: x[1], reverse = True) 
    
    distances = distances[1:6] # Get the 5 best scores , not including itself
    print(distances)
    
    # get the indices of the top 5
    recommendations = [distance[0] for distance in distances] 
    
    # return those recommendation names by pulling title from the given 5 indices
    return df['title'].iloc[recommendations] 


# In[15]:


import numpy as np

random_books_index = np.random.randint(low=0, high=len(df), size=1).astype(int)
print(random_books_index[0])

title = df["title"][random_books_index[0]]
print(title)

ContentBasedRecommender(title, indices, distance_matrix)


