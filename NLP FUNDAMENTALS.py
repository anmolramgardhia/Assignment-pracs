import numpy as np 

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, params):
    W1 , b1 , W2 , b2 = params
    z1 = x@ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    
    return z1 , h1 , z2

x_batch = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]])
params = (
    np.array([[0.2, -0.4], [0.7, 0.5], [-0.3, 0.8]]),  # W1
    np.array([0.1, -0.2]),                             # b1
    np.array([[0.6, -0.1], [-0.5, 0.9]]),              # W2
    np.array([0.0, 0.05]))                             # b2

z1, h1, z2 = forward_pass(x_batch, params)
print(z1 , h1 , z2) 

# Now we will implement the same thing using tensorflow
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X , y = make_moons(n_samples= 1000 , random_state=42 , learning_rate = 0.01)

X_train , X_test , y_train , y_test = train_test_split(X , y , stratify= y , random_state= 42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =  scaler.fit_transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(2 , ) , 
    tf.keras.layers.Dense(32 , activation= 'relu') , 
    tf.keras.layers.Dense(16 , activation= 'relu'),
    tf.keras.layers.Dense(1 , activation= 'sigmoid')
 ])

model.compile(Optimizer= tf.keras.optimizers.Adam(learning_rate = 0.01), 
              Loss = tf.keras.losses.BinaryCrossentropy() ,
               Metrics = ('accuracy'))

History = model.fit(X_train, y_train, batch_size=32, epochs=100,
                    validation_data=(X_test, y_test), verbose=0)

tf.keras.utils.plot_model(model, show_shapes=True)

# NLP QUE 1 

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Dataset
X, y = make_moons(n_samples=400, noise=0.25, random_state=7)
X = StandardScaler().fit_transform(X)
y = y.reshape(-1, 1)

# Init
np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros(3)
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros(1)
params = (W1, b1, W2, b2)

def relu(X):
    return np.maximum(0, X)

def forward_pass(X, params):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1          # ✅ (N, 3)
    h1 = relu(z1)              # ✅ (N, 3)
    z2 = h1 @ W2 + b2         # ✅ fixed order + correct bias
    y_pred = 1 / (1 + np.exp(-z2))  # ✅ sigmoid
    return z1, h1, z2, y_pred

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # numerical stability
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Forward
z1, h1, z2, y_pred = forward_pass(X, params)
loss = binary_cross_entropy(y, y_pred)

# Gradients
N = X.shape[0]
dz2 = (y_pred - y) / N   # BCE + sigmoid chain rule
dW2 = h1.T @ dz2
db2 = np.sum(dz2, axis=0)

print(y_pred[:5].flatten())
print(f"Loss: {loss:.6f}")
print(f"norm(dW2): {np.linalg.norm(dW2):.6f}")



# que 1 

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

model = None
vocab = {"<PAD>": 0, "<UNK>": 1}
MAX_LENGTH = 50

app = FastAPI()
def preprocess_text(text: str, vocab: Dict[str, int], max_length: int) -> torch.Tensor:
    text = text.lower()
    tokens = text.split()
    indices = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]

    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        padding = [vocab.get("<PAD>", 0)] * (max_length - len(indices)) 
        indices = indices + padding

    return torch.tensor([indices], dtype=torch.long)
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float ,

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_tensor = preprocess_text(request.text, vocab, MAX_LENGTH)
    with torch.no_grad():

        if model is None:
