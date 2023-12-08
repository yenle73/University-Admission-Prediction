import streamlit as st
import numpy as np
import pickle as pkl

st.title('University Admission Prediction')

input = open('lr_admit.pkl', 'rb')
model = pkl.load(input)

st.header('Input admission information')
gre = st.number_input('Insert GRE Score')
toefl = st.number_input('Insert TOEFL Score')
uni_rate = st.number_input('Insert University Rating')
sop = st.number_input('Insert SOP')
lor = st.number_input('Insert LOR')
cgpa = st.number_input('Insert CGPA')
re = st.radio('Choose research', [0, 1])

if re:
  if st.button('Predict'):
    feature_vector = np.array([gre, toefl, uni_rate, sop, lor, cgpa, re]).reshape(1,-1)
    result = str((model.predict(feature_vector)[0])[0])
    st.header('Result')
    st.write(result)
  




