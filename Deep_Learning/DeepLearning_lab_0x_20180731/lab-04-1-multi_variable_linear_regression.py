import tensorflow as tf
tf.set_random_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')

b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
print("hypothesis : " , hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val , hy_val , _  = sess.run( [cost , hypothesis , train] ,
                                       feed_dict={x1 : x1_data , x2: x2_data , x3: x3_data , Y: y_data})
    if step % 10  == 0:
        print(step , "Cost: " , cost_val , "\nPrediction: \n" , hy_val)

'''
 Cost:  62547.3 
Prediction: 
 [-75.96344757 -78.27629089 -83.83014679 -90.80435944 -56.97648239]
10 Cost:  14.4686 
Prediction: 
 [ 145.2640686   187.59541321  178.1519928   194.4858551   145.81135559]
 
 
 1990 Cost:  4.92225 
Prediction: 
 [ 148.1499939   186.88690186  179.62905884  195.81777954  144.4611969 ]
2000 Cost:  4.89701 
Prediction: 
 [ 148.15844727  186.88110352  179.63166809  195.8195343   144.45372009]
 
 20000 Cost:  0.215748 
Prediction: 
 [ 151.31877136  184.73472595  180.62504578  196.34005737  141.80513   ]
'''