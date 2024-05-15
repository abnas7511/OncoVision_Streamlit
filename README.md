# Streamlit_Breast-Cancer-Predictor
ML App for Breast Cancer Prediction using Streamlit 

This project implements a Breast Cancer Predictor using Machine Learning techniques. The application predicts whether a breast mass is benign or malignant based on various cell nuclei measurements.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Files Included](#files-included)
- [Usage](#usage)
- [Live Demo](#live-demo)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. This project aims to assist medical professionals in diagnosing breast cancer by providing a predictive model based on cell nuclei measurements.

## Technologies Used

- Python
- Streamlit
- scikit-learn
- Plotly
- Pandas

## Files Included

- `app.py`: Contains the Streamlit web application code. The application allows users to input cell nuclei measurements and receive real-time predictions on whether the tumor is benign or malignant.
- `model.py`: Implements the machine learning model using logistic regression. The model is trained on the Breast Cancer Wisconsin (Diagnostic) Dataset from Kaggle and achieves an accuracy of 97%.
- `data/data.csv`: Dataset containing cell nuclei measurements for training the model.
- `model/model.pkl`: Pickle file containing the trained machine learning model.
- `model/scaler.pkl`: Pickle file containing the scaler used for feature scaling during model training.

## Usage

To run the application locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit application using the command `streamlit run app.py`.
4. Access the application in your web browser at `http://localhost:8501`.

## Live Demo

Check out the live demo hosted on Streamlit: [Breast Cancer Predictor - Live Demo](https://breast-cancer-predictor-abnas7511.streamlit.app/)

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
