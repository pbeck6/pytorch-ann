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


    print(df.head())
#