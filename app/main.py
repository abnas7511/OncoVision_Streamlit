#to run this file you need to run like streamlit run main.py not like running python file
import os
import streamlit as st
import pickle as pickle
import pandas as pd


def get_clean_data():
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)
    
    # Construct the path to data.csv
    data_path = os.path.join(current_dir, '..', 'data', 'data.csv')
    
    # Load the data
    data = pd.read_csv(data_path)
    
    # Cleaning the data (to remove unnecessary attributes)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map mal = 1 and benign = 0 
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


#sidebar creation
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    #since we need the variable names and max,min values we need the data again so 
    data = get_clean_data()

    # Define the labels and column name
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    #to store values from the slider a dictionary
    input_dict = {}

    #loop through them and create a slider for each opf them
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= float(0),
            max_value= float(data[key].max()), #max value in the csv file for each columns
            value =  float(data[key].mean()) #setting default value
        ) 

    return input_dict


def main():
    st.set_page_config( #setting page config like page title,icon,layout and all
        page_title ="Breast Cancer Predictor",
        page_icon = "🦀",
        layout= "wide",
        initial_sidebar_state="expanded" #initially the sidebar will be expanded

    )

    input_data = add_sidebar()


    #div is mentioned here by container
    with st.container():
        st.title("Breast Cancer Predictor")#h1 tag
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")#p tag

    #creating columns
    col1,col2 = st.columns([4,1]) #the parameter is ratio of size first one will be 4 times greater than second one
    #since it reurns 2 cols it is saved into col1 and col2



    with col1:
        st.write("this is column 1")
    with col2:
        st.write("this is column 2")  



if __name__ == '__main__':
    main()