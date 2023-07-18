#!/usr/bin/env python
# coding: utf-8

# ### This is the raw dataset we are working with.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df


# # Personality Test-Taker (Randomizer)

# ### This step simulates how a user could possibly fill up the OCEAN personality test (i.e., what answer they could put in each question). It then outputs the correspoindng personality score calculations. 

# In[25]:


import pandas as pd
import random

final=[]

def big_five_personality_test():
    questions = {
        "Openness": [
            "I am someone who is always full of new ideas.",
            "I am quick to understand new things.",
            "I have high appreciation for art, culture, and literature.",
            "I am interested in abstract ideas.",
            "I am imaginative and creative."
        ],
        "Conscientiousness": [
            "I am organized and pay attention to details.",
            "I am diligent and complete tasks on time.",
            "I am reliable.",
            "I am NOT spontaneous.",
            "I am careful and cautious."
        ],
        "Extraversion": [
            "I am outgoing and sociable.",
            "I am talkative.",
            "I am energetic and enthusiastic.",
            "I take charge of projects and exhibit good leadership skills.",
            "I am not afraid of speaking in front of crowds."
        ],
        "Agreeableness": [
            "I am considerate and kind to others.",
            "I am competitive and like to win.",
            "I am cooperative and easy to get along with.",
            "I am respectful and not judgmental.",
            "I trust people easily."
        ],
        "Neuroticism": [
            "I get stressed out easily.",
            "I am anxious and worry a lot.",
            "I get nervous easily.",
            "I often have negative thoughts.",
            "I am moody and experience frequent mood swings."
        ]
    }

    scores = {
        "Openness": 0,
        "Conscientiousness": 0,
        "Extraversion": 0,
        "Agreeableness": 0,
        "Neuroticism": 0
    }
    result=[]
    
    for trait, trait_questions in questions.items():
        print("\nTrait:", trait)
        for i, question in enumerate(trait_questions):
            print("Question", i + 1, ":", question)
            while True:
                try:
                    answer = random.randint(1, 5)
                    print(answer)
                    if answer < 1 or answer > 5:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5.")
                    next

            scores[trait] += answer

    print("\nResults:")
    for trait, score in scores.items():
        max_score = 25
        percentage = convert_to_percentage(score, max_score)
        result.append(percentage)

    final.append(result)
    print(final)
    

def convert_to_percentage(score, max_score):
    """Converts a score to a percentage based on the given range."""
    percentage = (score) / (max_score)*100
    decimal = percentage/100
    return decimal

big_five_personality_test()


# ### The hypothetical user's simulated OCEAN score is then stored under the variable "final"

# In[26]:


final


# # Simulating OCEAN Scores and Cleaning

# ### Due to limitations, we simulated the "ideal" OCEAN score that is correlated with each song in our database.

# In[2]:


import pandas as pd
import numpy as np

filename='dataset.csv'
newsong=pd.read_csv(filename)

song = newsong[~newsong.duplicated(subset='track_name')].copy()
song.dropna(inplace=True)

# Assuming you have a DataFrame named 'song' with existing columns 'column1', 'column2', 'column3', and 'column4'

# Step 1: Generate correlated random numbers
n = len(song)  # Number of rows in the DataFrame
correlation_dict_O={'acousticness': -0.281, 'danceability': -0.014, 'duration_ms': 0.149, 'energy':0.263,'instrumentalness':-0.179,'liveness': 0.147,'loudness':0.139,'speechiness': 0.121,'tempo':0.086, 'valence':0.058}
correlation_dict_C={'acousticness': -0.001, 'danceability': -0.06, 'duration_ms': -0.045, 'energy':0.011,'instrumentalness':0.038,'liveness': 0.057,'loudness':0.029,'speechiness': -0.009,'tempo':0.004, 'valence':0.011}
correlation_dict_E={'acousticness': -0.019, 'danceability': -0.021, 'duration_ms': -0.056, 'energy':0.038,'instrumentalness':0.081,'liveness': 0.02,'loudness':0.056,'speechiness': -0.088,'tempo':0.01, 'valence':-0.115}
correlation_dict_A={'acousticness': -0.083, 'danceability': -0.081, 'duration_ms': -0.023, 'energy':0.073,'instrumentalness':0.081,'liveness': 0.079,'loudness':0.063,'speechiness': 0.046,'tempo':-0.052, 'valence':-0.012}
correlation_dict_N={'acousticness': 0.06, 'danceability': 0.069, 'duration_ms': 0.017, 'energy':-0.066,'instrumentalness':-0.013,'liveness': -0.019,'loudness':-0.063,'speechiness': 0.01,'tempo':-0.013, 'valence':0.035}

#OPENNESS

# Calculate the scaling factors based on the standard deviation
scaling_factors = {}
for column, correlation in correlation_dict_O.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_O)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Openness'] = scaled_numbers

#CONSCIENTIOUSNESS

scaling_factors = {}
for column, correlation in correlation_dict_C.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_C)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Conscientiousness'] = scaled_numbers

#EXTRAVERSION

scaling_factors = {}
for column, correlation in correlation_dict_E.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_E)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Extraversion'] = scaled_numbers

#AGREEABLENESS

scaling_factors = {}
for column, correlation in correlation_dict_A.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_A)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Agreeableness'] = scaled_numbers

#NEUROTICISM

scaling_factors = {}
for column, correlation in correlation_dict_N.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_N)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Neuroticism'] = scaled_numbers

# Print the updated DataFrame
song


# # Filtering System

# In[5]:


explicit_choice = input("Do you want us to include EXPLICIT songs in our recommendations? (Type 'Y' for yes and 'N' for no) ")
foreign_choice = input("Do you want us to include FOREIGN songs in our recommendations? (Type 'Y' for yes and 'N' for no) ")

genres_to_remove = ['afrobeat', 'brazilian', 'brazil', 'french', 'german', 'iranian', 'j-dance', 'j-idol',
                   'j-pop', 'j-rock', 'k-pop', 'latin', 'latino', 'malay', 'mandarin', 'mandopop', 'new-age',
                   'pagode', 'salsa', 'samba', 'spanish', 'sertanejo', 'swedish', 'tango', 'turkish', 'world-music']


if explicit_choice == "N" and foreign_choice == 'N':
    song = song.drop(song[song['explicit'] == True].index)
    song = song[~song['track_genre'].isin(genres_to_remove)]
    
elif explicit_choice == "Y" and foreign_choice == 'N':
    song = song[~song['track_genre'].isin(genres_to_remove)]
    
elif explicit_choice == "N" and foreign_choice == 'Y':
    song = song.drop(song[song['explicit'] == True].index)


# In[7]:


song


# # Machine Learning and Training

# ### In order to be able to provide a recommendation, we needed to use a regressor (i.e., XGBoost). As such, the data was first normalized since song attributes are not measured on the same scale.

# In[8]:


#Normalize Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


labels = ['duration_ms', 'danceability', 'energy',
       'key', 'loudness', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo']

normalized_test =scaler.fit_transform(song[labels])


# ### Here are the results of the normalization

# In[9]:


normalized_test


# ### Next, we used an XGBoost Regressor to train our model. The inputs were the "features" or OCEAN scores, while the outputs were the "labels" or song attributes. The XGB helps us predict what "outputs" are most likely to occur given a set of inputs (i.e., OCEAN personality score of a user)
# 
# ### Moreover, the group used SUPERVISED machine learning to train the model. This was done by using a train-test split.

# In[10]:


features = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness' , 'Neuroticism']


#Assigning x and y values
x = song[features] #Input
y = pd.DataFrame(normalized_test, columns = labels)  #Normalized Labels (Output)


#Trying Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


from xgboost import XGBRegressor

#XG Boost Regression + Train Test Split
xgb_regressor = XGBRegressor(random_state=42) #Creates bot/template
xgb = xgb_regressor.fit(x_train, y_train)


#OPTIONAL (below)
xgb_regression_score = xgb_regressor.score(x_test, y_test)


# # Recommending

# ### We then took the hypothetical OCEAN score of a user (under the variable "final") and used our "trained" XGB model to predict the "ideal" features of a song that a person of that personality would like.

# In[27]:


import numpy as np

#Make this random instead of inputs OR have a fixed example !!!

print("Loading Recommendations... Please Wait.")

user_personality = np.array(final)
user_personality = pd.DataFrame(user_personality, columns = features)

predictions = xgb.predict(user_personality)


# ### Here is the set of the user's hypothetical personality scores stored as a dataframe.

# In[15]:


user_personality


# ### This is the set of "ideal" song attribute values that are predicted to be associated with the user. For better visualization, we have shown these predictions as an array and a dataframe:

# In[16]:


predictions


# In[17]:


pd.DataFrame(predictions, columns = labels)


# ### From here, we used the method of cosine similarity to determine how SIMILAR each song (i.e., the song attributes of each song) were to the "ideal" predicted set of song attributes associated with our user.

# In[28]:


#Ranking the Similarity of Each Existing Score to the Predicted Score of the User
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


y_duplicate = y.copy()

array_vec_1 = np.array(predictions)
scores = []

for index, array in y.iterrows():
    
    score = cosine_similarity(array_vec_1, np.array([array]))  
    
    scores.append(score[0][0])
    

y_duplicate = y_duplicate.assign(score = scores)


# ### We then take the cosine similarity score of each song:

# In[19]:


y_duplicate


# ### From there, we sort our dataframe in descending order to get the songs with the HIGHEST similarity scores.

# In[29]:


y_duplicate = y_duplicate.sort_values('score', ascending = False)

y_duplicate


# ### We then get the indices of the top 5 songs:

# In[30]:


top_indices = y_duplicate.index.values.tolist()[0:5]

top_indices


# ### These indices are then cross-referenced with the original database in order to retrive the details of each song (i.e., song title and artist). These are then outputted to the user.

# In[31]:


print("Given Your Personality, You Might Like These Songs!")
counter = 1

for i in top_indices:
    print(counter, ":", song.iloc[i, 4], "by", song.iloc[i,2])
    counter+=1

