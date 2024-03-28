import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load data
data = pd.read_csv('E:\\UCM\\NueralNetworks\\Assignment-9\\Sentiment.csv')
data = data[['text', 'sentiment']]

# Preprocess data
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

max_features = 2000

# Tokenize text
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# Encode labels
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(data['sentiment'])

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define model architecture
def create_model(dropout_rate=0.2):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(196, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Custom wrapper class
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, dropout_rate=0.2):
        self.dropout_rate = dropout_rate
        self.model = create_model(dropout_rate=self.dropout_rate)
    
    def fit(self, X, y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X, y, callbacks=[early_stopping], epochs=1, batch_size=32, validation_split=0.2)
        return self
    
    def predict(self, X):
        return self.model.predict_classes(X)
    
    def set_params(self, **params):
        self.dropout_rate = params.get('dropout_rate', self.dropout_rate)
        self.model = create_model(dropout_rate=self.dropout_rate)
        return self
    
    def score(self, X, y):
        _, accuracy = self.model.evaluate(X, y, verbose=0)
        return accuracy
    
    def get_params(self, deep=True):
        return {'dropout_rate': self.dropout_rate}

# Define grid search parameters
param_grid = {
    'dropout_rate': [0.2, 0.3],
}

# Perform grid search
grid = GridSearchCV(estimator=KerasClassifierWrapper(), param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, Y_train)

# Print best parameters and best score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Save the model
grid_result.best_estimator_.model.save('model.h5')

# Load the model
loaded_model = load_model('model.h5')

# Predict on new data
new_text = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing. @realDonaldTrump"
new_text = re.sub('[^a-zA-Z0-9\s]', '', new_text.lower())
new_seq = tokenizer.texts_to_sequences([new_text])
new_pad_seq = pad_sequences(new_seq, maxlen=X.shape[1])
predicted_probabilities = loaded_model.predict(new_pad_seq)
predicted_class_index = predicted_probabilities.argmax(axis=-1)[0]
predicted_sentiment = labelencoder.inverse_transform([predicted_class_index])[0]
print('Predicted sentiment:', predicted_sentiment)