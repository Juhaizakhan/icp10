# icp10
This code performs sentiment analysis using a neural network model:

1. It loads text data and sentiments from a CSV file.
2. Preprocesses the text data by tokenizing and encoding sentiments.
3. Splits the data into training and testing sets.
4. Defines a neural network model with an embedding layer, LSTM layer, and softmax output layer.
5. Uses a custom wrapper class to integrate the model with scikit-learn's GridSearchCV.
6. Conducts hyperparameter tuning using grid search to find the best dropout rate.
7. Saves the best model to a file named 'model.h5'.
8. Loads the model back into memory.
9. Predicts the sentiment of new text data using the loaded model.

VIDEO LINK: [https://drive.google.com/file/d/1S49w0qtx7MThT4u7xKrO9gRcDsIzR35w/view?usp=sharing]
