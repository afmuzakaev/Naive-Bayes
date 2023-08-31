import numpy as np
from sklearn.naive_bayes import GaussianNB

# Generate random training data
np.random.seed(42)
num_samples = 100
X = np.random.rand(num_samples, 2)  # Features
y = np.random.choice([0, 1], size=num_samples)  # Class labels

# Train Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, y)

# Generate a random test input
test_input = np.random.rand(1, 2)

# Predict the label using the trained classifier
predicted_label = clf.predict(test_input)

print("Test Input:", test_input)
print("Predicted Label:", predicted_label[0])
