import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    print(f"TensorFlow version: {tf.__version__}")
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("TensorFlow basic matrix multiplication successful.")
except Exception as e:
    print(f"TensorFlow error: {e}")
