import numpy as np

np.random.seed(0)

for data_type in ['train', 'test']:
    with open(f"rank.{data_type}") as f:
        data = f.readlines()
    
    th1 = np.random.uniform(size=len(data))
    th2 = np.random.uniform(size=len(data))
    np.savetxt(f"rank.{data_type}.theta1", th1, delimiter="\n", fmt="%10.5f")
    np.savetxt(f"rank.{data_type}.theta2", th2, delimiter="\n", fmt="%10.5f")
