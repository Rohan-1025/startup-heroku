#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime


# In[16]:


st.write("""
# Startup future prediction app
""")

st.sidebar.header('User Input Features')

def user_input_features():
    country_code = st.sidebar.selectbox('Country Code',('USA','GBR','IND','CAN','DEU','AUS','FRA','ESP','NLD','ISR','IRL','BRA','other'))
    category_code = st.sidebar.selectbox('Category Code',('software','web','ecommerce','advertising','consulting','mobile','games_video','enterprise',
             'public_relations','network_hosting','hardware','education','search','biotech','other'))
    funding_total = st.number_input('Total funding (USD)')
    latitude = st.number_input('Latitude')
    longitude = st.number_input('Longitude')
    relationships = st.sidebar.slider('Relationship', 0,30,1)
    funding_rounds = st.sidebar.slider('Funding Rounds', 0,30,1)
    milestones = st.sidebar.slider('Milestones', -10,50,1)
    founded_at = st.date_input("Enter Date",min_value=datetime.date(1900, 1, 1),max_value=datetime.date(2023, 2, 2))

    data = {'category_code': category_code,
            'founded_at': founded_at,
            'country_code': country_code,
            'funding_rounds': funding_rounds,
            'funding_total_usd': funding_total,
            'milestones': milestones,
            'relationships': relationships,
            'lat': latitude,
            'lng': longitude
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
input_df["founded_at"] = pd.to_datetime(input_df["founded_at"], format = "%Y-%m-%d").dt.year


# In[17]:


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
companies_raw = pd.read_csv('clean_data.csv')
companies = companies_raw.drop(columns=['status','Unnamed: 0'])
df = pd.concat([input_df,companies],axis=0)

# Encoding of ordinal features

encode = ['country_code','category_code']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

st.write(input_df)
    
    
# Reads in saved classification model
log_reg = pickle.load(open('companies.pkl', 'rb'))

# Apply model to make predictions
prediction = log_reg.predict(df)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Operating')
else:
    st.write('Not-Operating')



# In[ ]:




