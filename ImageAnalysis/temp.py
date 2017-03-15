import numpy as np
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.optimizers import Adagrad

prediction=[[0,0,1,0],[0,1,0,0],[0,0,0,1]]
prediction = np_utils.categorical_probas_to_classes(prediction)

print (prediction)