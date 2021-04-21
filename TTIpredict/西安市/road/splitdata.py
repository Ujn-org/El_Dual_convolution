import pandas as pd
import numpy as np

data = pd.read_csv("西安市.txt").to_numpy()
pd.DataFrame(data[0:int(len(data)/24)]).to_csv("西安市_first_episoid.txt",header=None,index=None)