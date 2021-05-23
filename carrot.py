import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)


def Data_Genearion(num_points):
    vectors_set = []


    x_data = np.arange(0,20)
    y_data = np.array([-0.25,0.57,2.01,5.02,4.04,9.19,12.49,3.45,9.70,6.04,6.15,-7.32,-18.82,-11.77,-2.05,7.31,-7.86,-5.49,-4.30,-8.30])
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

    b = tf.Variable(tf.zeros([1]))

    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.00035)

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