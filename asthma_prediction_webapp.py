# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:39:18 2024

@author: nikitha
"""


import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open("C:/Users/nikitha/Downloads/asthma_pred_ml/asthma_model.sav",'rb'))

#creating a function for prediction
def asthma_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "The person has no asthma."
    else:
        return "The person has asthma."
    
def main():
    #giving a title
    st.title('Asthma Disease Prediction Web App')
   
    #getting the input from user
    Age=st.text_input('Enter the age')
    Sex=st.text_input('Sex(Female=0,Male=1)')
    Sleepingprob=st.text_input('Sleeping Problem (Yes=1, 0=No)')
    Chesttightness=st.text_input('Chest Tightness (Yes=1, 0=No)')
    Breath=st.text_input('Breathing Problem (Yes=1, 0=No)')
    Cough=st.text_input('Coughing Symptoms (Yes=1, 0=No)')
    Allergy=st.text_input('Allergies (Yes=1, 0=No)')
    Wheezing=st.text_input('Wheezing (Yes=1, 0=No)')
    
    
    #code for prediction 
    diagnosis=''
    
    #creating a button for prediction
    if st.button('Asthma Test Result'):
        diagnosis=asthma_prediction([Age,Sex,Sleepingprob,Chesttightness,Breath,Cough,Allergy,Wheezing])
        
    st.success(diagnosis)
    
if __name__ =='__main__':
    main()
    
    