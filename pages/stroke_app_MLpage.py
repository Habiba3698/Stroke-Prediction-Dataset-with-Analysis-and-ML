import streamlit as st
import joblib
import pandas as pd
inputs=joblib.load('input_columns.pkl')
pipeline_pre=joblib.load('preprocessing_pipeline.pkl')
pipeline=joblib.load('Stroke Prediction ML model.pkl')


def predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    test_df=pd.DataFrame(columns=inputs)
    test_df.at[0,'gender']=gender
    test_df.at[0,'age']=age
    test_df.at[0,'hypertension']=hypertension
    test_df.at[0,'heart_disease']=heart_disease
    test_df.at[0,'ever_married']=ever_married
    test_df.at[0,'work_type']=work_type
    test_df.at[0,'residence_type']=residence_type
    test_df.at[0,'avg_glucose_level']=avg_glucose_level
    test_df.at[0,'bmi']=bmi
    test_df.at[0,'smoking_status']=smoking_status

    x= pipeline_pre.transform(test_df)
    if pipeline.predict(x)[0] == 0:
        result = "You are not at risk for a stroke. But also remember to take care of your health and do regular checkups!"
    elif pipeline.predict(x)[0] == 1:
        result = "You are at risk for a stroke! Please see a doctor immediately!"
    return result
def main():
    st.image('stroke.png')
    st.title('🤖Stroke Prediction Based on Various Health and Lifestyle Factors')
    st.subheader("Let's say for person x: ")
    gender= st.selectbox('What is their gender?',['Male','Female'])
    age= st.slider('What is their age?',min_value= 1.0,max_value= 85.0,value=10.0,step=1.0)
    hypertension= st.selectbox('Do they have hypertension?',['0','1'])
    heart_disease= st.selectbox('Do they have Heart Disease?',['0','1'])
    ever_married= st.selectbox('Were they ever married?',['Yes','No'])
    work_type= st.selectbox('What is their work type? Are they children maybe?',['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type= st.selectbox('Were they ever married?',['Urban', 'Rural'])
    avg_glucose_level= st.slider('What is their average blood sugar level?',min_value= 1.0,max_value= 280.0,value=50.0,step=1.0)
    bmi= st.slider('What is their BMI?',min_value= 1.0,max_value= 100.0,value=20.0,step=1.0)
    smoking_status= st.selectbox('Do they currently smoke or have they ever smoked?',['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    
    
    if st.button('predict'):
        result=predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status)
        st.write(result)
if __name__ =='__main__':
    main()
