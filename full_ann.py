import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        """
        emb_szs: list[tuples]: (categorical variable size, embedding size)
        n_cont: int: # of continuous variables
        out_sz: int: output size
        layers: list[int]: layer sizes
        p: float: dropout probability for each layer
        """
        
        # Inherit from parent nn.Module
        super().__init__()
        
        # Set embedded layers in sequential order inside ModuleList container
        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        # Set dropout function to prevent overfitting for embeddings
        self.emb_drop = nn.Dropout(p)
        # Set up normalization function for continuous variables 
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Set up empty array to hold layers, size of combined cat and cont data
        layerlist = []
        n_emb = sum((nf for _, nf in emb_szs))
        n_in = n_emb + n_cont

        # Set up each layer of size i
        for i in layers:
            # Append linear layer of i nodes that processes n_in values (both cat and cont)
            layerlist.append(nn.Linear(n_in,i))
            # Add activation function
            layerlist.append(nn.ReLU(inplace=True))
            # Normalize continuous values
            layerlist.append(nn.BatchNorm1d(i))
            # Add dropout layer
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        # Layer to predict final value (regression as opposed to classification)
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Convert layers to Sequential container to chain layers together
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        """
        Preprocess embeddings and normalize cont variables before passing them through layers
        Use torch.cat() to concatenate multiple tensors into one
        """
        embeddings = []

        # Create tensor of embeddings for each feature (convert numerical cat (i) => embedding (e))
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i])) # every row, ith column (feature)

        # Combine embeddings into one 2d tensor (matrix) and preprocess with dropout func
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        # Preprocess continuous values with normalization func
        x_cont = self.bn_cont(x_cont)

        # Take in both cat and cont values row-by-row
        x = torch.cat([x, x_cont], 1) # concatenate over 1 dimension
        x = self.layers(x)

        # Output tensor with encodings for features
        return x

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    r = 6371  # Earth's average radius (km)
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # (km)

    return d

if __name__ == "__main__":
    # Read in dataset of taxi fares
    df = pd.read_csv('./data/nyctaxifares.csv')

    # Calculate trip distance from coordinates and add column to df
    # Alternate method using apply()
    # df['dist_km'] = df.apply(haversine_distance, axis=1, args=('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'))
    df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
    
    # Convert pickup datetime strings to datetime objs
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Adjust UTC datetime for EST in New York City
    df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)

    df['Hour'] = df['EDTdate'].dt.hour
    df['AMorPM'] = np.where(df['Hour']<12, 'am', 'pm')
    df['Weekday'] = df['EDTdate'].dt.strftime("%a")

    # Prepare categorical and continuous values
    cat_cols = ['Hour', 'AMorPM', 'Weekday']
    cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','passenger_count', 'dist_km']
    
    # Set y column as fare amount (the target value)
    y_col = ['fare_amount']

    # Change categorical values into Category objects with numerical code
    for cat in cat_cols:
        df[cat] = df[cat].astype('category')
    
    # Convert to array for use as PyTorch tensor
    hr = df['Hour'].cat.codes.values
    ampm = df['AMorPM'].cat.codes.values
    wkdy = df['Weekday'].cat.codes.values

    # Stack them column-wise like original data
    cats = np.stack([hr, ampm, wkdy], axis=1)

    # One-line alternate list comprehension for categorical values
    # cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)

    # Convert categorical data to PyTorch tensor
    cats = torch.tensor(cats, dtype=torch.int64)

    # Convert continuous values to PyTorch tensor
    conts = np.stack([df[col].values for col in cont_cols], axis=1)
    conts = torch.tensor(conts, dtype=torch.float)

    # Convert target label (taxi fare) into PyTorch tensor
    y = torch.tensor(df[y_col].values, dtype=torch.float)

    # Set embedding sizes (denser vector representation than one hot encoding)
    cat_szs = [len(df[col].cat.categories) for col in cat_cols]
    emb_szs = [(size, min(50,(size+1)//2)) for size in cat_szs]

    # Generate TabularModel obj
    # conts is a 2d tensor, conts.shape[1] = # of cols = # of cont features
    # emb_szs in this case will be size+1//2
    # Total # of in-features will be conts.shape[1] + sum([emb_szs[i][1] for i in emb_szs])
    torch.manual_seed(33)
    model = TabularModel(emb_szs, conts.shape[1], 1, [200,100], p=0.4) # out_sz = 1

    print(model)

    # Set loss function and optimization algorithm (alternative to stochastic gradient descent)
    criterion = nn.MSELoss() # np.sqrt(MSE) = RMSE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train-test split
    batch_size = 600
    test_size = int(batch_size*0.2)

    # Data came pre-shuffled, otherwise randomly shuffle beforehand
    cat_train = cats[:batch_size-test_size]
    cat_test = cats[batch_size-test_size:batch_size]
    con_train = conts[:batch_size-test_size]
    con_test = conts[batch_size-test_size:batch_size]

    y_train = y[:batch_size-test_size]
    y_test = y[batch_size-test_size:batch_size]

    # Train for # of epochs
    epochs = 30
    losses = []
    for i in range(epochs):
        i += 1

        # Forward pass
        y_pred = model(cat_train, con_train)

        # Track error
        loss = torch.sqrt(criterion(y_pred, y_train))
        losses.append(loss)

        if i%10 == 1:
            print(f'Epoch: {i} | Loss: {loss}')

        # Set zero gradient to erase accumulated weight/bias adjustments
        optimizer.zero_grad()
        # Backpropagation to create gradient for this pass
        loss.backward()
        # Adjust weights and biases according to gradient and learning rate
        optimizer.step()

    # Test neural net against values - no training means no gradient
    with torch.no_grad():
        y_val = model(cat_test, con_test)
        loss = torch.sqrt(criterion(y_val, y_test))
        print(f'Test loss is {loss}')

    # See sample of predicted vs actual values
    for i in range(10):
        print(f'Predicted: {y_val[i].item():8.2f} | Actual: {y_test[i].item():8.2f}')

    # Save neural net for future use, if satisfied
    # torch.save(model.state_dict(), 'taxi_model.pt')
#