# PeduliHIV-Machine-Learning

## Doctor Review & Hate Speech Filter Models
This repository contains two models developed using TensorFlow & Python in Jupyter colab Notebooks: the Hate Speech Filter Algorithm and the Doctor Review & Recommendation model. These models were trained using publicly available datasets from Kaggle that were already translated in the Indonesian Language.

## Dataset
The datasets used for training the models are as follows:
- Doctor Review Dataset: This dataset contains reviews on doctors, which were utilized to train the Doctor Review & Recommendation model. The model analyzes patient reviews to provide accurate doctor recommendations (https://www.kaggle.com/datasets/avasaralasaipavan/doctor-review-dataset-has-reviews-on-doctors ).
- Indonesian Abusive and Hate Speech Twitter Text: This dataset consists of Indonesian tweets containing abusive and hate speech. It was employed to train the Hate Speech Filter Algorithm model. The model detects hate speech by extracting contextual patterns (https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text ).

## Model Architecture
Both models utilize a combination of convolutional, dropout, and LSTM layers to capture relevant patterns and extract meaningful features from the input data, all available in the TensorFlow Library and Keras API
- The Hate Speech Filter Algorithm model employs TensorFlow's computational capabilities to optimize the model using gradient descent and backpropagation. It analyzes the contextual patterns within the text to classify whether it contains hate speech or not.
- The Doctor Review & Recommendation model leverages the TensorFlow framework to train on the doctor review dataset. By analyzing the patient reviews, it generates accurate recommendations for doctors based on the extracted insights.

## Acknowledgements
We would like to express our gratitude to the authors of the original datasets:
- Avasarala Sai Pavan for the Doctor Review Dataset.
- Ilham Fathoni Panigoro for the Indonesian Abusive and Hate Speech Twitter Text dataset.
We also extend our thanks to the TensorFlow team for providing the powerful framework and functions used for training the models.


