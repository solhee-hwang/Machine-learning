# -*- coding: utf-8 -*-
"""code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FerumP-iQpB4-jGfQTKhUz912VvAWNd2
"""

# 전체적으로 교수님 예제코드를 참고하여 모델링하였습니다.
# 4차 다항회귀
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 논벼 데이터
# vectors_set = [[0,-1.17],[1,2.60],[2,-6.34],[3,-12.31],[4,0.22],[5,-2.57],[6,-1.97],
#                [7,-7.34],[8,3.40],[9,6.18],[10,-3.96],[11,-1.37],[12,-5.95],[13,1.01],
#                [14,3.40],[15,7.77],[16,7.18],[17,4.79],[18,4.20],[19,2.01]]
# ------------
# 콩 데이터
# vectors_set = [[0,-20.99],[1,-9.53],[2,-14.35],[3,-20.99],[4, -1.69],[5,4.95],
#                [6,4.34],[7,-9.53],[8,6.15],[9,19.42],[10,-11.34],[11,0.12],
#                [12,-8.32],[13,16.41],[14,12.79],[15,10.37],[16,-7.12],
#                [17, 13.39],[18,6.76],[19,8.56]]
# ------------
# 고구마 데이터
# vectors_set = [[0,25.79],[1,26.44],[2,26.61],[3,11.77],[4,22.73],[5,-3.13],
#                [6,1.00],[7,-16.55],[8,-0.30],[9,-1.30],[10,-8.31],[11,-16.67],
#                [12,-12.25],[13,-12.61],[14,-7.54],[15,-10.37],[16,-13.20],
#                [17, -11.72],[18,-14.20],[19,-1.12]]
# -------------
# 배추데이터
vectors_set=[[0,37.79],[1,34.43],[2,7.38],[3,30.17],[4,21.45],[5,3.34],
             [6,15.41],[7,-5.03],[8,3.52],[9,-5.01],[10,-9.08],[11,8.36],
             [12,-19.41],[13,-11.50],[14,-10.69],[15,-23.47],[16,-30.11],
             [17,-9.15],[18,-10.91],[19,-26.75]]
# --------------
# 당근데이터
# vectors_set=[[0,-0.25],[1,0.57],[2,2.01],[3,5.02],[4,4.04],[5,9.19],
#              [6,12.49],[7,3.45],[8,9.70],[9,6.04],[10,6.15],[11,-7.32],
#              [12,-18.82],[13,-11.77],[14,-2.05],[15,7.31],[16,-7.86],
#              [17,-5.49],[18,-4.30],[19,-8.30]]
# --------------
# 자두데이터
# vectors_set=[[0,15.91],[1,15.70],[2,34.90],[3,27.26],[4,15.70],[5,20.26],
#              [6,13.90],[7,18.46],[8,22.49],[9,-9.43],[10,-16.11],[11,-24.70],
#              [12,-24.49],[13,-22.90],[14,-17.39],[15,-11.45],[16,-3.81],
#              [17,-9.12],[18,-21.52],[19,-23.64]]
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

def Data_Learning(x_data, y_data):
    W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    W3 = tf.Variable(tf.random_uniform([1], -1.0 ,1.0))
    W4 = tf.Variable(tf.random_uniform([1], -1.0 ,1.0))
    b = b = tf.Variable(tf.random_uniform([1],-20.0, 20.0))

    # y = W4*x_data*x_data*x_data*x_data + W3*x_data*x_data*x_data + W2*x_data*x_data + W1 * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000000003306)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in np.arange(70000):
        sess.run(train)
        if(step%10000==9999):
            print(step, sess.run(W4), sess.run(W3), sess.run(W2), sess.run(W1), sess.run(b))
            print(step, sess.run(loss))
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W4)*x_data*x_data*x_data*x_data + sess.run(W3)*x_data*x_data*x_data + sess.run(W2) *x_data * x_data + sess.run(W1) * x_data + sess.run(b))
            plt.xlim([-1,20])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()


if __name__ == '__main__':
    num_points=50
    Data_Learning(x_data, y_data)



# 전체적으로 교수님 예제코드를 참고하여 모델링하였습니다.
# 2차 다항회귀
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 논벼 데이터
# vectors_set = [[0,-1.17],[1,2.60],[2,-6.34],[3,-12.31],[4,0.22],[5,-2.57],[6,-1.97],
#                [7,-7.34],[8,3.40],[9,6.18],[10,-3.96],[11,-1.37],[12,-5.95],[13,1.01],
#                [14,3.40],[15,7.77],[16,7.18],[17,4.79],[18,4.20],[19,2.01]]
# ------------
# 콩 데이터
# vectors_set = [[0,-20.99],[1,-9.53],[2,-14.35],[3,-20.99],[4, -1.69],[5,4.95],
#                [6,4.34],[7,-9.53],[8,6.15],[9,19.42],[10,-11.34],[11,0.12],
#                [12,-8.32],[13,16.41],[14,12.79],[15,10.37],[16,-7.12],
#                [17, 13.39],[18,6.76],[19,8.56]]
# ------------
# 고구마 데이터
# vectors_set = [[0,25.79],[1,26.44],[2,26.61],[3,11.77],[4,22.73],[5,-3.13],
#                [6,1.00],[7,-16.55],[8,-0.30],[9,-1.30],[10,-8.31],[11,-16.67],
#                [12,-12.25],[13,-12.61],[14,-7.54],[15,-10.37],[16,-13.20],
#                [17, -11.72],[18,-14.20],[19,-1.12]]
# -------------
# 배추데이터
# vectors_set=[[0,37.79],[1,34.43],[2,7.38],[3,30.17],[4,21.45],[5,3.34],
#              [6,15.41],[7,-5.03],[8,3.52],[9,-5.01],[10,-9.08],[11,8.36],
#              [12,-19.41],[13,-11.50],[14,-10.69],[15,-23.47],[16,-30.11],
#              [17,-9.15],[18,-10.91],[19,-26.75]]
# --------------
# 당근데이터
# vectors_set=[[0,-0.25],[1,0.57],[2,2.01],[3,5.02],[4,4.04],[5,9.19],
#              [6,12.49],[7,3.45],[8,9.70],[9,6.04],[10,6.15],[11,-7.32],
#              [12,-18.82],[13,-11.77],[14,-2.05],[15,7.31],[16,-7.86],
#              [17,-5.49],[18,-4.30],[19,-8.30]]
# --------------
# 자두데이터
# vectors_set=[[0,15.91],[1,15.70],[2,34.90],[3,27.26],[4,15.70],[5,20.26],
#              [6,13.90],[7,18.46],[8,22.49],[9,-9.43],[10,-16.11],[11,-24.70],
#              [12,-24.49],[13,-22.90],[14,-17.39],[15,-11.45],[16,-3.81],
#              [17,-9.12],[18,-21.52],[19,-23.64]]
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

def Data_Learning(x_data, y_data):
    W1 = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([1],-1.0, 1.0)) 
    b = tf.Variable(tf.random_uniform([1],-20.0, 20.0)) 
    # x값이 0부터 시작하기 때문에 xeros대신 random_uniform으로 설정하였습니다.
    # 기본적으로 첫번째 데이터의 범위가 -20~20정도이기때문에 -20~20으로 설정했습니다.

    y = W2*x_data*x_data + W1*x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    # 학습률을 높이면 loss가 nan으로 떠 학습률을 낮추고 반복횟수를 높였습니다.
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

# 학습률을 낮추는 대신 반복횟수를 높여 loss값을 줄였습니다.
for step in np.arange(1000):
      if(step%100==99):
        sess.run(train)
        print(step, sess.run(W2), sess.run(W1), sess.run(b))
        print(step, sess.run(loss))
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W2) *x_data * x_data + sess.run(W1) * x_data + sess.run(b))
        plt.xlim([-1,20])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__ == '__main__':
    num_points=50
    Data_Learning(x_data, y_data)

# 전체적으로 교수님 예제코드를 참고하여 모델링하였습니다.
# 3차 다항회귀
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 논벼 데이터
# vectors_set = [[0,-1.17],[1,2.60],[2,-6.34],[3,-12.31],[4,0.22],[5,-2.57],[6,-1.97],
#                [7,-7.34],[8,3.40],[9,6.18],[10,-3.96],[11,-1.37],[12,-5.95],[13,1.01],
#                [14,3.40],[15,7.77],[16,7.18],[17,4.79],[18,4.20],[19,2.01]]
# ------------
# 콩 데이터
# vectors_set = [[0,-20.99],[1,-9.53],[2,-14.35],[3,-20.99],[4, -1.69],[5,4.95],
#                [6,4.34],[7,-9.53],[8,6.15],[9,19.42],[10,-11.34],[11,0.12],
#                [12,-8.32],[13,16.41],[14,12.79],[15,10.37],[16,-7.12],
#                [17, 13.39],[18,6.76],[19,8.56]]
# ------------
# 고구마 데이터
# vectors_set = [[0,25.79],[1,26.44],[2,26.61],[3,11.77],[4,22.73],[5,-3.13],
#                [6,1.00],[7,-16.55],[8,-0.30],[9,-1.30],[10,-8.31],[11,-16.67],
#                [12,-12.25],[13,-12.61],[14,-7.54],[15,-10.37],[16,-13.20],
#                [17, -11.72],[18,-14.20],[19,-1.12]]
# -------------
# 배추데이터
# vectors_set=[[0,37.79],[1,34.43],[2,7.38],[3,30.17],[4,21.45],[5,3.34],
#              [6,15.41],[7,-5.03],[8,3.52],[9,-5.01],[10,-9.08],[11,8.36],
#              [12,-19.41],[13,-11.50],[14,-10.69],[15,-23.47],[16,-30.11],
#              [17,-9.15],[18,-10.91],[19,-26.75]]
# --------------
# 당근데이터
# vectors_set=[[0,-0.25],[1,0.57],[2,2.01],[3,5.02],[4,4.04],[5,9.19],
#              [6,12.49],[7,3.45],[8,9.70],[9,6.04],[10,6.15],[11,-7.32],
#              [12,-18.82],[13,-11.77],[14,-2.05],[15,7.31],[16,-7.86],
#              [17,-5.49],[18,-4.30],[19,-8.30]]
# --------------
# 자두데이터
# vectors_set=[[0,15.91],[1,15.70],[2,34.90],[3,27.26],[4,15.70],[5,20.26],
#              [6,13.90],[7,18.46],[8,22.49],[9,-9.43],[10,-16.11],[11,-24.70],
#              [12,-24.49],[13,-22.90],[14,-17.39],[15,-11.45],[16,-3.81],
#              [17,-9.12],[18,-21.52],[19,-23.64]]
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


def Data_Learning(x_data, y_data):
    W1 = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    W3 = tf.Variable(tf.random_uniform([1], -1.0 ,1.0))
    b = b = tf.Variable(tf.random_uniform([1],-20.0, 20.0))

    y = W3*x_data*x_data*x_data + W2*x_data*x_data + W1 * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001306)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in np.arange(10000):
        sess.run(train)
        if(step%1000==999):
            print(step, sess.run(W3), sess.run(W2), sess.run(W1), sess.run(b))
            print(step, sess.run(loss))
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W3)*x_data*x_data*x_data + sess.run(W2) *x_data * x_data + sess.run(W1) * x_data + sess.run(b))
            plt.xlim([-1,20])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()


if __name__ == '__main__':
    num_points=50
    Data_Learning(x_data, y_data)