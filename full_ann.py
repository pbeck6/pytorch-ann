import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TabularModel(nn.Module):

    def __init__(self, emb_szs: list, n_cont, out_sz, layers, p=0.5):
        
        # Inherit from parent nn.Module
        super.__init__()
        
        # Set dense vector representation for categorical data
        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        # Set dropout layer to prevent overfitting
        self.emb_drop = nn.Dropout(p)
        # Normalize continuous data within some range
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum([nf for _, nf in emb_szs])
        n_in = n_emb + n_cont

        # Set up layers
        for i in layers:
            # Append linear layer
            layerlist.append(nn.Linear(n_in,i))
            # Add activation function
            layerlist.append(nn.ReLu(inplace=True))
            # Normalize continuous values
            layerlist.append(nn.BatchNorm1d(i))
            # Add dropout layer
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        # Predict final value
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist )

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        # Run categorical features through
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        # Normalize continuous values
        x_cont = self.bn_cont(x_cont)

        # Take in both cat and cont values row-by-row
        x = torch.cat([x, x_cont], axis=1)
        x = self.layers(x)

        # Output tensor with encodings for features
        return x

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

if __name__ == "__main__":
    # Read in dataset of taxi fares
    df = pd.read_csv('NYCTaxiFares.csv')

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

    # Convert categorical data to tensor
    cats = torch.tensor(cats, dtype=torch.int64)

    # Convert continuoouts values to numerical values in tensor
    conts = np.stack([df[col].values for col in cont_cols], axis=1)
    conts = torch.tensor(conts, dtype=torch.float)

    # Convert label into tensor
    y = torch.tensor(df[y_col].values, dtype=torch.float)

    # Set embedding sizes (denser vector representation than one hot encoding)
    cat_szs = [len(df[col].cat.categories) for col in cat_cols]
    emb_szs = [(size, min(50,(size+1//2))) for size in cat_szs]


#