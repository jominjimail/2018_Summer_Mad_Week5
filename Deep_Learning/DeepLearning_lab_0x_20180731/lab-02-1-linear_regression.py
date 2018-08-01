import tensorflow as tf

tf.set_random_seed(777)
# Variable 객체의 초기값은 보통 random으로 지정한다. 디버깅을 ㅏ거나 reproducibility를 위해서 random seed를 자주 사용한다.

x_train=[1,2,3]
y_train=[1,2,3]

W=tf.Variable(tf.random_normal( [1] ) , name='weight')
b=tf.Variable(tf.random_normal( [1] ) , name='bias')

hypothesis = x_train* W + b
cost = tf.reduce_mean( tf.square( hypothesis - y_train))
#실제 y_train 값과 Wx + b 의 값의 차이를 제곱해서 평균때린다. cost 가 적을수록 좋다.

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
#그 러닝 레이트가 다음학습할때 점프하는 거리랄까?
train = optimizer.minimize(cost)
#cost 를 가장 작게 하는게 목표지

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 200 ==0:
        print(step , sess.run(cost) , sess.run(W) , sess.run(b))


'''
1600 8.28196e-05 [ 1.01056981] [-0.02402747]
1800 3.16242e-05 [ 1.00653136] [-0.01484741]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
'''








