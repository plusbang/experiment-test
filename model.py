import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import time

from utils import distributed_train_keras, MultiprocessingBackend
# from bigdl.nano.common.multiprocessing.multiprocs_backend import MultiprocessingBackend

def create_data(tf_data=False, batch_size=32):
    train_num_samples = 1000
    test_num_samples = 400
    
    def get_x_y(num_sample):
        x = np.random.randn(num_sample)
        y = np.random.randn(num_sample)
        return x, y
    
    train_data = get_x_y(train_num_samples)
    test_data = get_x_y(test_num_samples)

    if tf_data:
        from_tensor_slices = tf.data.Dataset.from_tensor_slices
        train_data = from_tensor_slices(train_data).cache()\
                                                   .shuffle(train_num_samples)\
                                                   .batch(batch_size)\
                                                   .prefetch(tf.data.AUTOTUNE)
        test_data = from_tensor_slices(test_data).batch(batch_size)\
                                                 .cache()\
                                                 .prefetch(tf.data.AUTOTUNE)
    return train_data, test_data

class my_model():
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    def get_config(self):
        return self.model.get_config()
    
    def compile(self, learning_rate):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        def train(x, y):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss_value = self.loss_object(y, pred)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss_value
        
        @tf.function
        def train_opt(x, y):
            return train(x, y)

        def predict(x):
            return self.model(x, training=False)

        @tf.function
        def predict_opt(x):
            return predict(x)

        self.train_func = train_opt
        self.predict_func = predict_opt
        self.model.compile(optimizer=optimizer, loss=self.loss_object)
    
    def fit_batch(self, x):
        loss_value = self.train_func(x, y)
        return loss_value.numpy()
    
    def predict_batch(self, x):
        pred = self.predict_func(x)
        return pred.numpy()

train_data, test_data = create_data(tf_data=True)

model = my_model()
model.compile(0.001)
print(model.get_config())

_backend = MultiprocessingBackend()
fit_kwargs = dict(x=train_data)

st = time.time()
for i in range(0, 5000):
    #model.fit_batch(xs)
    res=distributed_train_keras(_backend, model.model, model.fit_batch, 3, fit_kwargs)
print("time spent: ", time.time()-st)
