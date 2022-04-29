import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simplest implementation of subclass of nn.Module
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        """
        in_features: int: how many initial features
        h1: int: in_features for fc1 (fully connected)
        h2: int: in_features for final activation layer
        out_features: int: in this case, size of one-hot encoding
        """
        super().__init__()

        # nn.Linear layer is simple transformation y = xA + b
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        """
        Forward propagation: accept input data, process per activation function, pass to successive layer
        """
        # Set activation function to rectified linear unit
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))

        return x

class TabularModel(nn.Module):
    """
    Model for dealing with tabular data using embeddings for vector representation
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    """

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

        # Set up empty array to hold layers, initial size of combined cat and cont data
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
            # out_features of this layer is in_features of next layer
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
