import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load the model
model=pickle.load(open('iris_model.pkl','rb'))

# app title
st.title('iris flower classification app')
st.write('predict the specis of iris flower using ml ')

# input fields
sepal_lenghth=st.number_input("sepal length (cm)",4.0,8.0,5.4)
sepal_width=st.number_input("sepal width (cm )",2.0,4.4,3.4)
petal_length=st.number_input('petal length(cm)',1.0,7.0,4.5)
petal_width=st.number_input('petal width (cm )',0.1,2.5,1.3)

# predict button
if st.button("predcit"):
  features=np.array([sepal_lenghth,sepal_width,petal_length,petal_width])
  prediction=model.predict([features])
  species=['Setosa','versicolor','virginica']
  st.subheader("Prediction:")
  st.write(species[prediction[0]])
