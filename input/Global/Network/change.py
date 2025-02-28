import numpy as np
import pandas as pd
df = pd.read_csv("mrio.csv",header=None)
df = df.map(lambda x: np.random.randint(0, 11) if x == 0 else x)
print(df)
df.to_csv("mrio2.csv", index=False, header=False)