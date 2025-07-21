
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title='Heart Stroke  Prediction App', layout='wide')


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_stroke_data.csv')

df = load_data()

st.title("📊 Exploratory Data Analysis - Heart Stroke Data")

tab1, tab2 = st.tabs(["Univariate & Bivariate Analysis", "Multivariate Analysis"])

with tab1:
    st.subheader("📈 Univariate Analysis and 🔁 Bivariate Analysis")
    # side bar 
    x = st.sidebar.checkbox('Show Data', False, key=1)
    stroke_filter = st.sidebar.selectbox("Select Stroke", np.append(df['stroke'].unique(),'All'))
 
    # show data if button is chosen                              
    if x:

        st.header('DataSet Sample')
        st.dataframe(df.sample(100))

    selected_feature = st.sidebar.selectbox("Select Feature for Analysis", df.columns[:-1])

    # Filtered Data
    if stroke_filter != 'All':
        df_filtered = df[df['stroke'] == int(stroke_filter)]
    else:
        df_filtered = df

    
    counts_df = df_filtered[selected_feature].value_counts().reset_index(name='no_of_individuals')
    counts_df['stroke']= df_filtered['stroke']
    

    if pd.api.types.is_numeric_dtype(df_filtered[selected_feature]):
                # show outliers
        q1 = df_filtered[selected_feature].quantile(0.25)
        q3 = df_filtered[selected_feature].quantile(0.75)
        iqr = q3 - q1
        outliers = df_filtered[(df_filtered[selected_feature] < q1 - 1.5 * iqr) | (df_filtered[selected_feature] > q3 + 1.5 * iqr)]
        st.write(f" Outliers count: {outliers.shape[0]}")
        st.plotly_chart(px.box(df_filtered, y=selected_feature))
        st.write(f"### Distribution of {selected_feature}")
        st.dataframe(counts_df)

        st.plotly_chart(px.histogram(df_filtered, x=selected_feature, color='stroke', color_discrete_sequence=px.colors.qualitative.Vivid))
    else:
        st.warning("Selected column is not numeric. Can't calculate statistics")
        st.write(f"### Distribution of {selected_feature}")
        st.dataframe(counts_df)

        st.plotly_chart(px.histogram(df_filtered, x=selected_feature, color='stroke', color_discrete_sequence=px.colors.qualitative.Vivid))

    # bivariate analysis but stroke rate for each case not counts 
    st.subheader("Bivariate Analysis with stroke rate (not counts)")
    
    # display the rate of each stroke case with feature
    grouped= df.groupby([selected_feature, 'stroke'], dropna=False).size().reset_index(name='no_of_individuals')   
    grouped['percent'] = grouped.groupby(selected_feature)['no_of_individuals'].transform(lambda x: x / x.sum() * 100)
    
    if stroke_filter != 'All':
        grouped = grouped[grouped['stroke'] == int(stroke_filter)]
    else:
         grouped=grouped
        
    

    if pd.api.types.is_numeric_dtype(df_filtered[selected_feature]):
        if(selected_feature== 'bmi' or selected_feature=='avg_glucose_level' or selected_feature=='age'):
            st.write(f"### Average {selected_feature} by Stroke Group")
            avg = df.groupby('stroke')[selected_feature].mean().reset_index()   
            if stroke_filter != 'All':
                avg = avg[avg['stroke'] == int(stroke_filter)]
            else:
                 avg=avg
                
            st.dataframe(avg)      
            st.plotly_chart(px.bar(avg, x='stroke', y=selected_feature, color=avg['stroke'].astype(str), barmode='group', color_discrete_sequence=px.colors.qualitative.Vivid))  

        
        st.write(f"### Rate of stroke with {selected_feature}")
        st.dataframe(grouped)
        st.plotly_chart(px.bar(grouped, x=selected_feature, y='percent', color=grouped['stroke'].astype(str), barmode='group', color_discrete_sequence=px.colors.qualitative.Vivid))
        st.write(f"### Relationship of stroke with {selected_feature}")
        st.plotly_chart(px.strip(df_filtered, x='stroke', y=selected_feature, color='stroke', stripmode='overlay', color_discrete_sequence=px.colors.qualitative.Vivid))
    else:
        st.write(f"### Rate of stroke with {selected_feature}")
        st.dataframe(grouped)
        st.plotly_chart(px.bar(grouped, x=selected_feature, y='percent' , color=grouped['stroke'].astype(str), barmode='group', color_discrete_sequence=px.colors.qualitative.Vivid)) 
        
with tab2:
        st.subheader("🔀 Multivariate Analysis")
        numeric_cols = df.select_dtypes(include='number').columns
        

        y = st.sidebar.checkbox('Show Heatmap', False, key=2)
            # show data if button is chosen                              
        if y:

            st.header('Heatmap')
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='Blues', square=True, ax=ax)
            st.pyplot(fig)
        selection = st.selectbox("Please select the information you wish to see (all features are against stroke rate)",
                                 [  "Age and BMI",
                                    "Lifestyle factors: Smoking and Work Type",
                                    "Health factors: Blood Glucose and Heart Disease",
                                    "Gender and Hypertension",
                                    "Blood Glucose and Hypertension",
                                    "Combined Risk: Blood Glucose, Hypertension, and Smoking"])
        if selection == "Age and BMI":
            st.subheader("Age and BMI vs Stroke Rate")
            grouped= df.groupby("stroke")[["age","bmi"]].mean() 
            st.dataframe(grouped)
            fig=px.scatter(df, x='age',y='bmi', color=df['stroke'].astype(str), title='BMI vs Age by Stroke', color_discrete_sequence=px.colors.qualitative.Vivid, trendline='lowess')
            fig.update_layout(legend_title_text='Stroke') 
            st.plotly_chart(fig, use_container_width=True)

        elif selection == "Lifestyle factors: Smoking and Work Type":
            st.subheader("Lifestyle Factors (Smoking & Work Type) vs Stroke Rate")
            grouped = df.groupby(['work_type', 'smoking_status', 'stroke']).size().reset_index(name='no_of_individuals')
            grouped['percent'] = grouped.groupby(['work_type', 'smoking_status'])['no_of_individuals'].transform(lambda x: x / x.sum() * 100)
            st.dataframe(grouped)
            fig=px.bar(grouped, height=500, width=3500, x='work_type', y='percent', color=grouped['stroke'].astype(str), barmode='group', 
                       hover_data=['no_of_individuals'], facet_col='smoking_status', color_discrete_sequence=px.colors.qualitative.Vivid,
                      title="Work type and stroke rate (both 0 and 1) facet by smoking staus", labels={'percent': 'Stroke Rate (%)', 'color': 'stroke'})
            
            st.plotly_chart(fig, use_container_width=True)

        elif selection == "Health factors: Blood Glucose and Heart Disease":
            st.subheader("Blood Glucose & Heart Disease vs Stroke Rate")
            heart= df.groupby(["stroke","heart_disease"])["avg_glucose_level"].mean().reset_index()  
            st.dataframe(heart)
            fig= px.bar(heart, x='heart_disease', y='avg_glucose_level', color=heart['stroke'].astype(str), 
                   barmode='group', color_discrete_sequence=px.colors.qualitative.Vivid,
                   title='Average Glucose by heart disease and Stroke',
                   labels={'avg_glucose_level': 'Average glucose','color':'stroke'})
            st.plotly_chart(fig, use_container_width=True)

        elif selection == "Gender and Hypertension":
            st.subheader("Gender & Hypertension vs Stroke Rate (positive only)")
            stroke_rate = (df.groupby(['gender', 'hypertension'])['stroke'].mean().reset_index())
            stroke_rate['stroke_rate_percent'] = stroke_rate['stroke'] * 100
            stroke_rate['Hypertension Status'] = stroke_rate['hypertension'].map({0: 'No Hypertension', 1: 'Hypertension'})
            st.dataframe(stroke_rate)
            fig= px.bar(
                stroke_rate,x='gender',y='stroke_rate_percent',color='Hypertension Status',barmode='group',
                labels={'stroke_rate_percent': 'Stroke Rate (%)', 'gender': 'Gender'},
                title='Stroke Rate by Gender and Hypertension',
                color_discrete_map={'No Hypertension': 'royalblue', 'Hypertension': 'crimson'}
            )

            st.plotly_chart(fig, use_container_width=True)

        elif selection == "Blood Glucose and Hypertension":
            st.subheader("Blood Glucose & Hypertension vs Stroke Rate")
            rate= df.groupby(["stroke","hypertension"])["avg_glucose_level"].mean().reset_index()    
            st.dataframe(rate)
            fig= px.bar(rate, x='hypertension', y='avg_glucose_level', color=rate['stroke'].astype(str), 
                           barmode='group', color_discrete_sequence=px.colors.qualitative.Vivid,
                           title='Average Glucose by Hypertension and Stroke', labels={'avg_glucose_level': 'Average glucose','color':'stroke'})
            st.plotly_chart(fig, use_container_width=True)

        elif selection == "Combined Risk: Blood Glucose, Hypertension, and Smoking":
            st.subheader("Combined Risk Factors vs Stroke Rate")
            glucose = df.groupby(['hypertension', 'smoking_status'])[['avg_glucose_level', 'stroke']].mean().reset_index()
            glucose['stroke_rate_percent'] = glucose['stroke'] * 100
            st.dataframe(glucose)
            plot = pd.melt(glucose, id_vars=['hypertension', 'smoking_status'], value_vars=['avg_glucose_level', 'stroke_rate_percent'],
            var_name='Metric',value_name='Value')
            fig= px.bar(
                        plot,x='hypertension',y='Value',color='Metric',barmode='group', facet_col='smoking_status',
                        height=500, width=1500, labels={'stroke_rate_percent': 'Stroke Rate (%)'},
                        title='Stroke Rate and average blood sugar by hypertension and facet smoking status ',
                        color_discrete_map={'avg_glucose_level': 'royalblue', 'stroke_rate_percent': 'crimson'}
                        )

            st.plotly_chart(fig, use_container_width=True)
            
