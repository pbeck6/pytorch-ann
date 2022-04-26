import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

########################################
class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()

        # Fully connected neural networks
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        

    """
    Forward propagation, 
    Accept input data, process per activation function, pass to successive layer
    """
    def forward(self, x):
    
        # Set activation function to rectified linear unit
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))

        return x
########################################

# set seed for initializing random weights & biases    
torch.manual_seed(4555)
# print seed
print(torch.initial_seed())
model = Model()

df = pd.read_csv('iris.csv')

# Matplotlib plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()
# End matplotlib

X = df.drop('target', axis=1) # pandas dataframe
y = df['target'] # pandas series

X = X.values # convert to numpy array
y = y.values

# organize data into sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33) 

# convert features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# convert labels to long tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# criteria to measure how far off predictions are from data aka loss function
criterion = nn.CrossEntropyLoss()

# optimization algorithm will adjust weights and biases according to learning rate
# model.parameters() will feed in description of the layers of NN for Adam to optimize
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# epoch = 1 run through all the training data
epochs = 100

# keep track of losses
losses = []

for i in range(epochs):
    # run forward through network and get prediction
    y_pred = model.forward(X_train)

    # calculate the loss (first run will be very bad)
    # using CrossEntropyLoss requires no one-hot encoding aka [[1,0,0], [0,1,0], [0,0,1]]
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # print every 10th epoch
    if i%10 == 0:
       print(f'Epoch {i} and loss is {loss}')

    # backpropagation to adjust weights and biases
    # first reset all gradients to zeros to erase previous adjustments
    optimizer.zero_grad()
    # calculate gradient (vector of partial derivatives of target function with respect to input variables)
    loss.backward()
    # tell optimization algorithm to adjust parameters(weights/biases) based on gradient (step towards local minima based on first derivative of cost/loss function)
    optimizer.step()

# plot loss over epoch
plt.plot(range(epochs), losses)
plt.ylabel('LOSS')
plt.xlabel('Epoch')
plt.show()

# turn off autograd, no gradient needed for testing since we will not do backpropagation afterwards
with torch.no_grad():
    # send test values through network layers to get output (predictions)
    y_eval = model.forward(X_test)
    # send predictions through loss function (comparing X_train:y_train, X_test:y_test)
    loss = criterion(y_eval, y_test)
    print(f'Loss of test values {loss}')

# count how many predictions were correct
correct = 0
with torch.no_grad():
    # go through each piece of data
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        # predictions are printed in probabilities w/ one-hot encoding
        print(f'{i+1}.) {str(y_val)} {y_test[i]}')

        # see if prediction was correct
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'{correct} correct predictions')


# save model file to disk, essentially the learned parameters(layers) of the model
torch.save(model.state_dict(), 'my_iris_model.pt')

# load model, assumes that Model class still exists
new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))
# check model was loaded
new_model.eval()

# try classifying a single new flower (should be indexs 0, iris setosa)
mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
with torch.no_grad():
    # send mystery flower through network layers to get output (prediction)
    print(new_model.forward(mystery_iris))