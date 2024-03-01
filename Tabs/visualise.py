"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
'''from sklearn.metrics import plot_confusion_matrix'''
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
import streamlit as st


# Import necessary functions from web_functions
from web_functions import train_model

def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Pneumonia Level")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize = (10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)



    if st.checkbox("Display Boxplot"):
        fig, ax = plt.subplots(figsize=(15,5))
        df.boxplot(["Age","Gender","Body_Temperature","Cough","Sore_Throat","Difficulty_Breathing","Chest_Pain","White_Blood_Cell_Count","Heart_Rate","Respiratory_Rate"],ax=ax)
        st.pyplot()

    # if st.checkbox("Show Sample Results"):
    #     safe = (df['Pneumonia'] == 1).sum()
    #     low = (df['Pneumonia'] == 2).sum()
    #     med = (df['Pneumonia'] == 3).sum()
    #     high = (df['Pneumonia'] == 4).sum()
    #     vhigh = (df['Pneumonia'] == 5).sum()
    #     data = [safe,low,med,high,vhigh]
    #     labels = ['Safe', 'Low','Medium','High','Very High']
    #     if (df['Pneumonia'] == 1).any():
    #         print("Pneumonia")
    #     elif(df['Pneumonia']==0):
    #         print("not Pneumonia")
    # Create a checkbox to show sample results
    if st.checkbox("Show Sample Results"):
        # Count the number of occurrences for each stage
        safe = (df['Pneumonia'] == 1).sum()
        low = (df['Pneumonia'] == 2).sum()
        med = (df['Pneumonia'] == 3).sum()
        high = (df['Pneumonia'] == 4).sum()
        # vhigh = (df['Pneumonia'] == 5).sum()
        data = [safe, low, med, high]
        labels = ['Safe', 'Low', 'Medium', 'High'] 

        # Check for the presence of Pneumonia
        if (df['Pneumonia'] == 1).any():
            st.write("Pneumonia")
        elif (df['Pneumonia'] == 0).all():
            st.write("Not Pneumonia")
        # colors = sns.color_palette('pastel')[0:5]
        # # plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        # st.pyplot()
        # Create a figure with a smaller size
        fig, ax = plt.subplots(figsize=(2, 2))
        colors = sns.color_palette('pastel')[0:5]
        ax.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        st.pyplot(fig)

    

    
    
