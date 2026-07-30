# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, train, and evaluate a simple Artificial Neural Network (ANN) using PyTorch to learn the relationship between a single numeric input and a single numeric output from a given dataset. After training, the model should be able to predict the output for new input values and display the training loss versus iteration graph to show how the model learns over time.

## Neural Network Model

<img width="1102" height="737" alt="image" src="https://github.com/user-attachments/assets/6a5e33d5-9699-4060-82b8-2f1e94b23888" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:RAGASUDHA R

### Register Number:212224230215

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim


# Load Dataset
data = pd.read_csv("Exp-1 (1).csv")

print("===== DATASET INFORMATION =====")
print(data)

print("\nShape:", data.shape)

print("\nStatistical Summary:")
print(data.describe())


# Input and Output
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Normalize Dataset
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# Convert to Tensor
xt = torch.FloatTensor(X_train)
yt = torch.FloatTensor(y_train)

xtest = torch.FloatTensor(X_test)
ytest = torch.FloatTensor(y_test)


# Neural Network
class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


# Model
model = NeuralNet()

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training
epochs = 1000

losses = []

for i in range(epochs):

    optimizer.zero_grad()

    pred = model(xt)

    loss = criterion(pred, yt)

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.6f}")


# Plot Loss
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Training Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# Test Model
model.eval()

with torch.no_grad():

    y_pred = model(xtest)

    test_loss = criterion(y_pred, ytest)

print("\nTest Loss:", test_loss.item())


# Prediction
new_input = np.array([[50]])

new_input_scaled = x_scaler.transform(new_input)

with torch.no_grad():

    prediction_scaled = model(torch.FloatTensor(new_input_scaled))

prediction = y_scaler.inverse_transform(prediction_scaled.numpy())

print("\n===== NEW SAMPLE DATA PREDICTION =====")
print("Input Value :", new_input[0][0])
print("Predicted Output :", prediction[0][0])
```

### Dataset Information
<img width="672" height="642" alt="image" src="https://github.com/user-attachments/assets/ae8dceb4-c995-4bf3-bc75-7e489280a292" />

### OUTPUT
<img width="747" height="442" alt="image" src="https://github.com/user-attachments/assets/38ddd6f7-05e3-45a9-9f9f-09a3621c64a0" />

### Training Loss Vs Iteration Plot
<img width="1142" height="597" alt="image" src="https://github.com/user-attachments/assets/f3438c21-fb79-4fb9-a0d0-521195af6e3d" />


### New Sample Data Prediction
<img width="607" height="80" alt="image" src="https://github.com/user-attachments/assets/e98232a2-cddb-4eb5-8ff5-d373f3df4604" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
