import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Creates negative shuffles and splits training sets.

data_file = sys.argv[1]
binding_x_data = np.load("../data/"+data_file)

shuffled_negative_x_data = [np.random.permutation(x) for x in binding_x_data]

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((binding_x_data, shuffled_negative_x_data)),
                                                    np.concatenate((np.full((binding_x_data.shape[0], 2),[0, 1]),
                                                                    np.full((binding_x_data.shape[0], 2), [1, 0]))), test_size=0.1)

np.savez("./data/"+data_file[:-4]+".npz", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(data_file[:-4] + ".npz saved to \"data\"!")
