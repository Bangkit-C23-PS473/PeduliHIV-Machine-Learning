# Doctor Review Sentiment Analysis Model

This notebook is a sentiment analysis on various doctor reviews (In Indonesian) that mostly revolve around after appointing with the doctor, and also some about HIV consultations. The reviews only range from either a positive (1) review, or a negative (0) review.


## Dataset
The model is trained on a labeled dataset of text samples, where each sample is classified as either a negative doctor review, or a positive doctor review in Indonesian.. The dataset is not included in this repository, but you can find similar datasets on platforms like Kaggle or by conducting a web search.

## Python Libraries
We use various libraries for this model, which are similiar to the Hate Speech Filter Model. Pandas is used for extracting and reading the data, numpy is for mathematical calculations and array formation, matplotlib for plotting, and TensorFlow for utilizing the various layers and methods for the machine learning model which include the Tokenizer API for encoding and doing Sentiment Analysis. 


## Model Architecture
The hate speech detection model is built using the following layers:
1. Dropout Layer: Dropout is applied to the input data to reduce overfitting. It randomly sets a fraction of input units to 0 during training.
2. Dense Layer: A dense layer with a specified number of units is added after the dropout layer. It helps in learning complex patterns from the input data.
3. Convolutional Layer: This layer performs convolution operations on input (this being text that has been encoded) data. This layer will extract the features from the encoded data and feed it through the next layers.
4. GlobalMaxPooling Layer: GlobalMaxPooling is a pooling operation commonly used in convolutional neural networks (CNNs) for down-sampling and dimensionality reduction. It reduces the spatial dimensions of the input data while preserving its features.
5. Output Layer: The output layer is a dense layer with a sigmoid activation function, which produces a probability score between 0 and 1. This indicates the likelihood of a text sample being classified as a positive or a negative review.

## Usage
To use the hate speech detection model, follow these steps:
1. Download the dataset from Kaggle and save it to your Google Drive:
   - Go to the Kaggle website (https://www.kaggle.com/datasets/avasaralasaipavan/doctor-review-dataset-has-reviews-on-doctors)  and navigate to the dataset you want to download.
   - Click on the "Download" button to download the dataset file.
   - Upload the downloaded dataset file to your Google Drive.
2. Manually label the data:
   - Open the dataset file you downloaded from Kaggle.
   - Assign a label of 1 to text samples that is a positive review and a label of 0 to text samples that are a negative review.
   - Save the labeled dataset.
   Note: Ensure that the labeled dataset is in a format that can be processed by the hate speech detection model.

3. Load the trained model: Load the trained model weights or architecture from the repository. If you have trained the model locally, you can upload the model files to your workspace.
```bash
from google.colab import drive
drive.mount('/content/drive')
os.chdir("/content/drive/Shareddrives/CAPSTONE BANGKIT 2023/Machine Learning/Doctor (ID) Recommendation Sentiment Analysis")
```

4. Run all the programs provided in the IPYNB files:
   - Open the IPYNB files in a Jupyter Notebook environment or a similar Python IDE.
   - Run all the cells in the IPYNB files in the given order. This will preprocess the data, train the hate speech detection model, and evaluate its performance.

5. Save the trained model:
   - After running all the cells in the IPYNB files, you can save the trained model using the following code:

   ```python
   model.save('doctor_review.h5')
   ```
   
## Dependencies

The following dependencies are required to run the hate speech detection model:

- Python (version >= 3.6)
- TensorFlow (version >= 2.0)
- Keras (version >= 2.0)
- NumPy (version >= 1.0)
- Pandas (version >= 1.0)
