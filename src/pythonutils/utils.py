import numpy as np

# one-hot encoding
def one_hot_encode2(y, n_classes):
    # Convert pandas Series or DataFrame to numpy array
    if hasattr(y, 'values'):
        y = np.asarray(y.values, dtype=np.int64)
    else:
        y = np.asarray(y, dtype=np.int64)
    
    # Initialize the one-hot encoded matrix
    res = np.zeros((len(y), n_classes))
    
    # Fill in the 1s
    for i in range(len(y)):
        res[i, y[i]] = 1
        
    return res
