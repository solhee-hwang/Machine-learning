import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)


def Data_Genearion(num_points):
    vectors_set = []


    x_data = np.arange(0,20)

    #배추
    y_data = np.array([37.79,34.43,7.38,30.17,21.45,3.34,15.41,-5.03,3.52,-5.01,-9.08,8.36,-19.41,-11.50,-10.69,-23.47,-30.11,-9.15,-10.91,-26.75])
    #당근
    #y_data = np.array([-0.25,0.57,2.01,5.02,4.04,9.19,12.49,3.45,9.70,6.04,6.15,-7.32,-18.82,-11.77,-2.05,7.31,-7.86,-5.49,-4.30,-8.30])
    #자두
    #y_data = np.array([15.91,15.70,34.90,27.26,15.70,20.26,13.90,18.46,22.49,-9.43,-16.11,-24.70,-24.49,-22.90,-17.39,-11.45,-3.81,-9.12,-21.52,-23.64])
    #논벼
    #y_data=np.array([-1.17,2.60,-6.34,-12.31,0.22,-2.57,-1.97,-7.34,3.40,6.18,-3.96,-1.37,-5.95,1.01,3.40,7.77,7.18,4.79,4.20,2.01])
    #콩
    #y_data=np.array([-20.99,-9.53,-14.35,-20.99, -1.69,4.95,4.34,-9.53,6.15,19.42,-11.34,0.12,-8.32,16.41,12.79,10.37,-7.12,13.39,6.76,8.56])
    #고구마
    #y_data=np.array([25.79,26.44,26.61,11.77,22.73,-3.13,1.00,-16.55,-0.30,-1.30,-8.31,-16.67,-12.25,-12.61,-7.54,-10.37,-13.20, -11.72,-14.20,-1.12])

    return  x_data, y_data



def Data_Draw(x_data, y_data):
    plt.plot(x_data, y_data,'ro')
    plt.ylim([-50,50])
    plt.xlim([0,20])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def Data_Learning(x_data, y_data):
    W = tf.Variable(tf.random_uniform([1], -1.0 ,1.0))

    b = tf.Variable(tf.random_uniform([1],-50,50))

    y = W * x_data + b


    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.00055)

    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    for step in np.arange(50):
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