# Hate Speech Filter Detection Model
This repository contains a machine learning model for hate speech detection. The model is built using dropout, dense, and LSTM layers, which are commonly used in natural language processing tasks.

## Dataset
The model is trained on a labeled dataset of text samples, where each sample is classified as either hate speech or non-hate speech. The dataset is not included in this repository, but you can find similar datasets on platforms like Kaggle or by conducting a web search.

## Model Architecture
The hate speech detection model is built using the following layers:
1. Dropout Layer: Dropout is applied to the input data to reduce overfitting. It randomly sets a fraction of input units to 0 during training.
2. Dense Layer: A dense layer with a specified number of units is added after the dropout layer. It helps in learning complex patterns from the input data.
3. LSTM Layer: Long Short-Term Memory (LSTM) is a type of recurrent neural network layer that can capture the sequential nature of text data. It is used to learn the contextual information and dependencies within the text.
4. Output Layer: The output layer is a dense layer with a sigmoid activation function, which produces a probability score between 0 and 1. It indicates the likelihood of a text sample being classified as hate speech.

## Usage
To use the hate speech detection model, follow these steps:
1. Download the dataset from Kaggle and save it to your Google Drive:
   - Go to the Kaggle website (https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text ) and navigate to the dataset you want to download.
   - Click on the "Download" button to download the dataset file.
   - Upload the downloaded dataset file to your Google Drive.
2. Manually label the data:
   - Open the dataset file you downloaded from Kaggle.
   - Assign a label of 1 to text samples that contain hate speech and a label of 0 to text samples that do not contain hate speech.
   - Save the labeled dataset.
   Note: Ensure that the labeled dataset is in a format that can be processed by the hate speech detection model.

3. Load the trained model: Load the trained model weights or architecture from the repository. If you have trained the model locally, you can upload the model files to your workspace.
```bash
from google.colab import drive
drive.mount('/content/drive')
os.chdir("/content/drive/Shareddrives/CAPSTONE BANGKIT 2023/Machine Learning/Hate Speech 2")
```

4. Run all the programs provided in the IPYNB files:
   - Open the IPYNB files in a Jupyter Notebook environment or a similar Python IDE.
   - Run all the cells in the IPYNB files in the given order. This will preprocess the data, train the hate speech detection model, and evaluate its performance.

5. Save the trained model:
   - After running all the cells in the IPYNB files, you can save the trained model using the following code:

   ```python
   model.save('hate_speech_model.h5')
   ```
   
## Dependencies

The following dependencies are required to run the hate speech detection model:

- Python (version >= 3.6)
- TensorFlow (version >= 2.0)
- Keras (version >= 2.0)
- NumPy (version >= 1.0)
- Pandas (version >= 1.0)
