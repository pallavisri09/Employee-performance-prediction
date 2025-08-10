import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")



st.title("Employee Performance Prediction")

st.divider()

st.write("You can get a performance estimation for employee after entering the values and pressing to the predict button")

st.divider()

years = st.number_input("Enter the years at company",min_value=0, max_value=15, value=2)
salary = st.number_input("Enter monthly salary", min_value=1000, max_value=100000, value=5000)
overtime = st.number_input("Enter overtime hours",min_value=0,max_value=100,value=0)
promotions = st.number_input("Enter promotions", min_value=0,max_value=10, value=0)
satisfaction = st.number_input("Enter employee satisfaction",min_value=0.0,max_value=5.0,value=2.0)

X = [years, salary, overtime, promotions, satisfaction] 

st.divider()

predictionbutton = st.button("Product the performance score!")

st.divider()

if predictionbutton:
    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    
    st.balloons()
    
    st.write(f"Prediction for the performance score is {prediction}")


else:
    st.write("Please use the button for the prediction")