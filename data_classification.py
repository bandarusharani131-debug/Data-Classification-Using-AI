# AI Project 2
# Data Classification Using Decision Tree

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris = load_iris()

# Features (X) and Target (y)
X = iris.data
y = iris.target

# Display dataset information
print("Dataset Loaded Successfully")
print("Number of Samples:", len(X))
print("Number of Features:", len(X[0]))
print("Target Classes:", iris.target_names)
print()

# Split dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))
print()

# Create Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print()

# Classification Report
print("Classification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("New Sample:", sample)
print("Predicted Flower:", iris.target_names[prediction][0])