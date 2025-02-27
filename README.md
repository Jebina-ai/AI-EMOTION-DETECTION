# AI-EMOTION-DETECTION

Emotion Classification Using SVM

Overview

This project implements an emotion classification system using a Support Vector Machine (SVM) model. The dataset consists of tweets labeled with emotions such as happiness, sadness, surprise, and anger. The main objective is to preprocess the text data and train an SVM model to classify tweets based on their emotional content.

Dataset

The dataset used in this project is text_emotion.csv, which contains the following columns:

tweet_id: Unique identifier for each tweet (removed during preprocessing)

emotion: The emotion label associated with the tweet (target variable)

author: The author of the tweet (removed during preprocessing)

content: The tweet text (input feature)

Preprocessing Steps

Dropped unnecessary columns (tweet_id and author).

Converted tweet text to lowercase.

Removed retweets (rt).

Replaced @username mentions with an empty string.

Removed punctuation.

Removed links.

Removed extra tabs and spaces.

Trimmed blank spaces at the beginning and end of the text.

Data Splitting

The dataset is split into training (70%) and testing (30%) sets using train_test_split from sklearn.model_selection.

Feature Extraction

The text data is transformed into a numerical representation using vectorization (vect.transform).

The transformed data is stored as a sparse matrix.

Model Training

The model is trained using a Support Vector Machine (SVM) classifier (SVC) with an RBF kernel.

The random_state is set to 1 for reproducibility.

Model Evaluation

Predictions are made on the test set.

The accuracy score is calculated using metrics.accuracy_score.

The achieved accuracy is 63.49%.

Dependencies

pandas

scikit-learn

numpy

Usage

Ensure all dependencies are installed.

Run the script to preprocess data, train the model, and evaluate performance.

Modify the model parameters or preprocessing steps for further improvements.

Future Enhancements

Experiment with different feature extraction techniques (TF-IDF, word embeddings, etc.).

Try other machine learning models (Naive Bayes, Random Forest, Neural Networks).

Perform hyperparameter tuning to optimize model performance.
