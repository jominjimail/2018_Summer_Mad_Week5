import tensorflow as tf

tf.set_random_seed(777)

X=[1,2,3]
Y=[1,2,3]

W = tf.Variable(100.0)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10000):
    print(step , sess.run(W))
    sess.run(train)

