# Simple LSTM Chatbot Model for Tourism

This model is a simple LSTM chatbot that can answer questions related to tourism in Egypt. It is designed to be used as a proof-of-concept or as a starting point for more complex models.

## Data

The data used to train this model consists of a set of tourism-related questions and their corresponding answers in addition to some questions related to other subjects. The data was manually collected and preprocessed before being used to train the model.

## Model Architecture

The model is built using a recurrent neural network with LSTM cells. It has an embedding layer that maps words to a lower-dimensional vector space, followed by a single LSTM layer, and finally a fully connected layer that produces the output.

## Training

The model was trained using the Adam optimizer and a cross-entropy loss function. The data was split into training and validation sets, and early stopping was used to prevent overfitting.

## Evaluation

The results showed that the model performed well on the test set and was able to answer questions related to tourism with a moderate degree of accuracy.

## Conclusion

This simple LSTM chatbot model for tourism provides a starting point for building more complex models. It demonstrates the potential of using deep learning techniques to build chatbots that can answer questions and provide information on a specific domain.
