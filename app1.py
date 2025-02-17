# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:17:36 2025

@author: Dell
"""

#importing reuired libraries
import streamlit as st
import joblib

#loading our vectorization and GBC model saved files) 
vectorizer = joblib.load("vectorization.jb")
model = joblib.load('./GBC.jb')

#Title for the website
st.title("Real or Fake news Analysis")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Real and Fake News Analysis app </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

#Subtitle
st.write("Enter a news Article below")

#User inputs 
news_input = st.text_area("News Article:","")

#create a button to check news is real or fake
if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input]) #transforming the input given by user
        prediction = model.predict(transform_input) #Transormed input to the model for the prediction
        
        
        if prediction[0]==1:
            st.success("The News is Real ")
        else:
            st.error("The News is Fake ")
    else:
        st.warning("Please enter some text to analize. ")
        
if st.button("About"):
    st.text("This app will help us to get Real or Fake news")