# pytorch-ann

## Artificial neural network for regression
---
### About pytorch-ann

![View screen](https://www.fast.ai/images/kaggle_taxi.png)

This PyTorch project uses the simple neural net model used by the third place winners of the [Kaggle Taxi Trip Duration Competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) and applies the model to the problem of predicting cab fares. The model was first built with one-hot encoding and trained using the oft-used [iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set), and later re-trained using the taxi trip data and embedded layers for denser vector representations.

### How to run
After installing [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), run the following in a conda prompt:
```
conda env create -f pytorchenv.yml
conda activate pytorchenv
python test.py
```
If everything is working, the output should be a 1d tensor (5x3) of random ints. 

With the environment set up, try running the neural network:
`python full_ann.py`

Note: the constants in full_ann.py (EPOCHS, BATCH_N) are deliberately set to low numbers by default to run on a variety of machines. To obtain better (but slower) results and lower loss, try increasing BATCH_N.
