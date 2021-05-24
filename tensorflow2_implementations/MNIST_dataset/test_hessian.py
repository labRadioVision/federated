import tensorflow as tf
from tensorflow import keras

x = tf.random.normal([7, 5])
inputs = tf.keras.layers.Input(shape=(7, 5))
layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(inputs)
layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)(layer1)
model = keras.Model(inputs=inputs, outputs=layer2)
with tf.GradientTape(persistent=True) as t2:
  with tf.GradientTape(persistent=True) as t1:
    # x = layer1(x)
    # x = layer2(x)
    x = model(x)
    loss = tf.reduce_mean(x**2)
  #kk = layer1.kernel
  g = t1.gradient(loss, model.trainable_variables)
  num_train_layers = len(model.trainable_variables)
  ssd = model.trainable_variables[0][1,2]
hessian_v = []
for k in range(num_train_layers):
    hessian_v.append(t2.jacobian(g[k], model.trainable_variables)) # one row-block of the hessian
t=1