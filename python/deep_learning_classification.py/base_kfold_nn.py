import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

print(tf.config.list_physical_devices())