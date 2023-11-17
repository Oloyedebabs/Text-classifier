import pandas as pd
#import librries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
#load dataset

df=pd.read_csv("dataset-2.csv")
# Split dataset to feature X and label y
X = df['text']
Y = df['sentiment']
# Split dataset to train and test (80% train 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Vectorize text data

vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(X_train)
x_test_vectorized = vectorizer.transform(X_test)

# Model training
#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_vectorized, Y_train)
knn_prediction = knn.predict(x_test_vectorized)

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train_vectorized, Y_train)
dt_prediction = dt.predict(x_test_vectorized)

# Model evaluation
#KNN
def evaluate_model(y_true, y_pred, model_name):

    accuracy = accuracy_score(y_true, y_pred)
    precision= precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Performance {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("\n")

#Evaluate model
evaluate_model(Y_test, knn_prediction, "K-NN")

#Evaluate Decison tree

evaluate_model(Y_test, dt_prediction, "Decision Tree")

  



