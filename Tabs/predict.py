"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import pandas as pd
# Import necessary functions from web_functions
# from web_functions import predict
import web_functions


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Prediction of Pneumonia Type and Intensity.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

   
    st.subheader("Select Values:")
    
    col1,col2= st.columns(2)
    with col1:
        # Take input of features from the user.
        Age = st.slider("Age", df["Age"].min(), df["Age"].max())
      
        Gender = st.slider("Gender", df["Gender"].min(), df["Gender"].max())    
        Body_Temperature = st.slider("Body_Temperature", float(df["Body_Temperature"].min()), float(df["Body_Temperature"].max()))
        Cough = st.slider("Cough", int(df["Cough"].min()), int(df["Cough"].max()))
        Sore_Throat = st.slider("Sore_Throat", int(df["Sore_Throat"].min()), int(df["Sore_Throat"].max()))
       

    with col2:
         Difficulty_Breathing = st.slider("Difficulty_Breathing", float(df["Difficulty_Breathing"].min()), float(df["Difficulty_Breathing"].max()))
         Chest_Pain = st.slider("Chest_Pain", int(df["Chest_Pain"].min()), int(df["Chest_Pain"].max()))
         White_Blood_Cell_Count = st.slider("White_Blood_Cell_Count", float(df["White_Blood_Cell_Count"].min()), float(df["White_Blood_Cell_Count"].max()))
         Heart_Rate = st.slider("Heart_Rate", int(df["Heart_Rate"].min()), int(df["Heart_Rate"].max()))
         Respiratory_Rate = st.slider("Respiratory_Rate", int(df["Respiratory_Rate"].min()), int(df["Respiratory_Rate"].max()))


   
         feature_values = [
    Age,
    Gender,
    Body_Temperature,
    Cough,
    Sore_Throat,
    Difficulty_Breathing,
    Chest_Pain,
    White_Blood_Cell_Count,
    Heart_Rate,
    Respiratory_Rate
]
         
    
    # Create a button to predict
    # if st.button("Predict"):
    #     # Get prediction and model score
    #     prediction, score = web_functions.predict(X, y, features)
    if st.button("Predict"):
        prediction, score = web_functions.predict(X, y, feature_values)
    # Get prediction and model score
    # Pass the actual feature values for prediction, not the column names
    # prediction, score = web_functions.predict(X, y, feature_values)  # Fixed line

    # ... [rest of your code below]    
     

        if (Age< 60):
            st.info("Follow clinical procedures and recommendations with 600 mg of paracetamol to keep away fever")

        if (Age >60):
            st.info("Immediate attention needed!")

        # Print the output according to the prediction
        if (prediction == 1):
            st.warning("The person has risk of Aspiration Pneumonia")
            st.info("Severity Level 1: This is a nominal pneumonia and gets cured easily.")
            st.success("Smell some Eucalyptus oil and inhale medicated vapour especially with clove oil")

        elif (prediction == 2):
            st.warning("The person has risk of Bacterial Pneumonia")
            st.info("Severity Level 2: This is a common pneumonia and requires some mild doses of medication.")
            st.success("Requires medical attention and nebulizaton and medication courses like antibiotics and antihismatics. Consult a Physician for more details.")

        elif (prediction == 3):
            st.error("The person has high risk of Viral Pneumonia")
            st.info("Severity Level 3: This is a severe pneumonia and needs good medical attention and proper course of medication")
            st.success("Required ventilation / air purifier and stronger doses of antibiotics. However it gets cured faster than bacterial pneumonia.")

        elif (prediction == 4):
            st.error("The person has Fungal Pneumonia")
            st.info("Severity Level :4 This is a chronic pneumonia and is hard to cure if medications and clinical care are not provided in time.")
            st.success("Require Amycline or similar antibiotics and possible chances of being admitted to ICU with ventilation. Requires hospital level treatment.")
       
        # Print teh score of the model 
        st.sidebar.info("The model used is trusted by doctor and has an accuracy of " + str((score*100)) + "%")

        st.sidebar.markdown('''<a href="https://www.drugs.com/medical-answers/antibiotics-treat-pneumonia-3121707/
" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: orange; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 10px;">Best Medication for Pneumonia</a>''',unsafe_allow_html=True)