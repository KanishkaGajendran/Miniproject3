import streamlit as st
import pandas as pd
import numpy as np
import pickle 

rf_model = pickle.load(open('rf_model.pkl', 'rb'))
dt_model = pickle.load(open('dt_model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

st.set_page_config(page_title="DS_Employee Attrition Analysis and Prediction")
st.title("📊 Employee Attrition & Job Satisfaction Prediction")

menu =st.sidebar.selectbox("Selecy Model",
                           ["Employee Attrition",
                            "Performance Rating"])

if menu == "Employee Attrition":
    st.subheader(" 🔍 Predict Employee Attrition")

    Age=st.number_input("Age",18,60)
    Department=st.selectbox("Department",["Sales","Research & Development","Human Resources"])
    MonthlyIncome=st.number_input("MonthlyIncome",min_value=1000)
    JobSatisfaction=st.selectbox("JobSatisfaction",[1,2,3,4])
    YearsAtCompany=st.number_input("YearsAtCompany",min_value=0)
    MaritalStatus=st.selectbox("MaritalStatus",["Single","Married","Divorced"])
    OverTime=st.selectbox("OverTime",["Yes","No"])

    #Encoding
    Department_map={'Sales':0,'Research & Development':1,'Human Resources':2}
    Marital_map={'Single':0,'Married':1,'Divorced':2}
    OverTime_map={'Yes':1,'No':0}
    
    if st.button("Predict Attrition"):
        data=np.array([[Age,
                        Department_map[Department],
                        MonthlyIncome,
                        JobSatisfaction,
                        YearsAtCompany,
                        Marital_map[MaritalStatus],
                        OverTime_map[OverTime]]])
        
        data_scaled=scaler.transform(data) 
        prediction=rf_model.predict(data_scaled)[0]
        probability=rf_model.predict_proba(data_scaled)[0][1]


        if prediction == 1:
            st.error(f"⚠️ Employee Likely to Leave (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Employee Likely to Stay (Probability: {1-probability:.2f})")

elif menu == "Performance Rating":

    st.subheader("📈 Predict Employee Performance Rating ")

    Education=st.selectbox("Education Level",[1,2,3,4,5])
    JobInvolvement=st.selectbox("JobInvolvement",[1,2,3,4])
    JobLevel=st.selectbox("JobLevel",[1,2,3,4,5])
    MonthlyIncome=st.number_input("MonthlyIncome",min_value=1000)
    YearsAtCompany=st.number_input("YearsAtCompany",min_value=0)
    YearsInCurrentRole=st.number_input("YearsInCurrentRole",min_value=0)
    YearsSinceLastPromotion=st.number_input("YearsSinceLastPromotion")
    TrainingTimesLastYear=st.number_input("TrainingTimesLastYear")
    
    
    input_data=np.array([[Education,
                          JobInvolvement,
                          JobLevel,
                          MonthlyIncome,
                          YearsAtCompany,
                          YearsInCurrentRole,
                          YearsSinceLastPromotion,
                          TrainingTimesLastYear]])

    if st.button("Predict Performance"):
        prediction=dt_model.predict(input_data)[0]
    
        st.success(f"⭐ Predicted Performance Rating:{prediction}")     
                               
                                 
