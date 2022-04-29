import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get basic neural net Model
from neuralnet.models import Model

if __name__ == "__main__":

    # Set seed for initializing random weights & biases    
    torch.manual_seed(4555)

    # Initialize new Model obj
    model = Model()

    # Read in csv as pandas DataFrame
    df = pd.read_csv('./data/iris.csv')

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

    # Build dataset from df
    X = df.drop('target', axis=1) # pandas dataframe
    y = df['target'] # pandas series

    # Convert to numpy array
    X = X.values 
    y = y.values

    # Organize data into sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33) 

    # Convert features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Convert labels to long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Criteria to measure how far off predictions are from data aka loss function
    criterion = nn.CrossEntropyLoss()

    # Optimization algorithm will adjust weights and biases according to learning rate
    # model.parameters() will feed in description of the layers of NN for Adam to optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Epoch = 1 run through all the training data
    epochs = 100

    # Keep track of losses
    losses = []

    for i in range(epochs):
        # Run forward through network and get prediction
        y_pred = model.forward(X_train)

        # Calculate the loss (first run will be very bad)
        # Using CrossEntropyLoss requires no one-hot encoding aka [[1,0,0], [0,1,0], [0,0,1]]
        loss = criterion(y_pred, y_train)
        losses.append(loss)

        # Print every 10th epoch
        if i%10 == 0:
            print(f'Epoch {i} and loss is {loss}')

        # Backpropagation to adjust weights and biases
        # First reset all gradients to zeros to erase previous adjustments
        optimizer.zero_grad()
        # Calculate gradient (vector of partial derivatives of target function with respect to input variables)
        loss.backward()
        # Tell optimization algorithm to adjust parameters(weights/biases) based on gradient (step towards local minima based on first derivative of cost/loss function)
        optimizer.step()

    # Plot loss over epoch
    plt.plot(range(epochs), losses)
    plt.ylabel('LOSS')
    plt.xlabel('Epoch')
    plt.show()

    # Turn off autograd, no gradient needed for testing since we will not do backpropagation afterwards
    with torch.no_grad():
        # Send test values through network layers to get output (predictions)
        y_eval = model.forward(X_test)
        # Send predictions through loss function
        loss = criterion(y_eval, y_test)
        print(f'Loss of test values {loss}')

    # Count how many predictions were correct
    correct = 0
    with torch.no_grad():
        # Go through each piece of data
        for i, data in enumerate(X_test):
            y_val = model.forward(data)
            # Predictions are printed in probabilities w/ one-hot encoding
            print(f'{i+1}.) {str(y_val)} {y_test[i]}')

            # See if prediction was correct
            if y_val.argmax().item() == y_test[i]:
                correct += 1

    print(f'{correct} correct predictions out of {len(X_test)}')


    # Optional code block for saving, loading, and running optimized model
    """
    # Save model file to disk, essentially the learned parameters(layers) of the model
    torch.save(model.state_dict(), 'iris_model.pt')

    # Load model, assumes that Model class still exists
    new_model = Model()
    new_model.load_state_dict(torch.load('iris_model.pt'))
    # Check model was loaded
    new_model.eval()

    # Try classifying a single new flower (should be index 0, iris setosa)
    mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
    with torch.no_grad():
        # Send mystery flower through network layers to get output (prediction)
        print(new_model.forward(mystery_iris))

    """
