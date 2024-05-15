#to run this file you need to run like streamlit run main.py not like running python file

import streamlit as st
import pickle as pickle
import pandas as pd


def main():
    st.set_page_config( #setting page config like page title,icon,layout and all
        page_title ="Breast Cancer Predictor",
        page_icon = "🦀",
        layout= "wide",
        initial_sidebar_state="expanded"

    )

    st.write("hello world!")


if __name__ == '__main__':
    main()