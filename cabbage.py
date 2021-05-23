import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)


def Data_Genearion(num_points):
    vectors_set = []


    x_data = np.arange(0,20)
    y_data = np.array([37.79,34.43,7.38,30.17,21.45,3.34,15.41,-5.03,3.52,-5.01,-9.08,8.36,-19.41,-11.50,-10.69,-23.47,-30.11,-9.15,-10.91,-26.75])
    return  x_data, y_data



def Data_Draw(x_data, y_data):
    plt.plot(x_data, y_data,'ro')
    plt.ylim([-50,50])
    plt.xlim([0,20])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def Data_Learning(x_data, y_data):
    W = tf.Variable(tf.random_uniform([1], -1.5 ,0.0))
    b = tf.Variable(tf.random_uniform([1], 40.0,40.0))
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.00065)

    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    for step in np.arange(20):
        sess.run(train)
        print(step, sess.run(W), sess.run(b))
        print(step, sess.run(loss))
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    num_points=50
    x_data, y_data=Data_Genearion(num_points)
    Data_Draw(x_data, y_data)
    Data_Learning(x_data, y_data)