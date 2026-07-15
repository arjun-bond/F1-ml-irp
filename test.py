import pandas as pd

test = pd.Series([1, 2, 3, 1, 5])
print(test == 1)
print((test == 1).astype(int))