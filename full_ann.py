import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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