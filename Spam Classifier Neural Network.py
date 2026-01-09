import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

emails = [
    "Get rich quick! Click here now", 
    "Hey, are we still meeting for lunch?",
    "Free entry to win a prize tonight",
    "Please find the attached invoice for your order",
    "WINNER! You have won a thousand dollars",
    "Can you send me those notes from class?"
]

labels = np.array([[1, 0, 1, 0, 1, 0]]).T 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()

np.random.seed(1)
input_nodes = X.shape[1]
hidden_nodes = 4
output_nodes = 1


weights_0_1 = np.random.normal(0.0, 0.1, (input_nodes, hidden_nodes))
weights_1_2 = np.random.normal(0.0, 0.1, (hidden_nodes, output_nodes))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.5
for epoch in range(1000):
    
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, weights_0_1))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
    
    
    layer_2_error = labels - layer_2
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    
    layer_1_error = layer_2_delta.dot(weights_1_2.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
    
    
    weights_1_2 += layer_1.T.dot(layer_2_delta) * learning_rate
    weights_0_1 += layer_0.T.dot(layer_1_delta) * learning_rate


test_email = ["Free money now"]
test_vector = vectorizer.transform(test_email).toarray()
prediction = sigmoid(np.dot(sigmoid(np.dot(test_vector, weights_0_1)), weights_1_2))

print(f"Spam Probability: {prediction[0][0]:.4f}")

print("Result:", "Spam" if prediction > 0.5 else "Ham")



Ouput:-

Spam Probability: 0.9015
Result: Spam

