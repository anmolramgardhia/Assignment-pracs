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
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X , y = make_moons(n_estimators = 1000 , learning_rate = 0.01 , random_state=42)
X_train , X_test , y_train , y_test = train_test_split(X , y , stratify = y , random_state= 42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.inputs(shape=(2 , )) ,
    tf.keras.layers.Dense(32 , activation = 'relu') , 
    tf.keras.layers.Dense(16 , activation = 'relu') , 
    tf.keras.layers.Dense(1 , activation = 'sigmoid')
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
              Loss = tf.keras.Losses .BinaryCrossentropy() ,
               metrics = ['accuracy'] )

model.utils.plotmodels(model , show_shapes = True)