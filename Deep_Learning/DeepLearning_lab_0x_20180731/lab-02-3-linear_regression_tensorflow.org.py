import tensorflow as tf

W = tf.Variable( [.3] , tf.float32)
b= tf.Variable ( [-.3] , tf.float32)

x= tf.placeholder(tf.float32)
y= tf.placeholder(tf.float32)

linear_model = x*W + b
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x: x_train , y:y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s , b: %s , loss: %s" % (curr_W, curr_b, curr_loss))

'''
W: [-0.21999997] , b: [-0.456] , loss: 4.01814
W: [-0.39679998] , b: [-0.49552] , loss: 1.81987

W: [-0.99999684] , b: [ 0.9999907] , loss: 5.77707e-11
W: [-0.9999969] , b: [ 0.99999082] , loss: 5.69997e-11
'''