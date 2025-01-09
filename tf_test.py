import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Test a basic TensorFlow operation
a = tf.constant(5)
b = tf.constant(3)
c = a + b
print("TensorFlow test result:", c.numpy())
