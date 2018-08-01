import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32 , shape=[None , 3])
Y= tf.placeholder(tf.float32 , shape = [None , 1])

W = tf.Variable(tf.random_normal( [3,1]) , name='weight')
b= tf.Variable(tf.random_normal( [1]) , name = 'bias')

hypothesis = tf.matmul( X , W ) + b
cost = tf.reduce_mean(tf.square((hypothesis - Y)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(200001):
    cost_val , hy_val , _ = sess.run(
        [cost , hypothesis , train] , feed_dict={X:x_data , Y:y_data})
    if step % 10 ==0:
        print(step , "Cost : " , cost_val , "\nPrediction: \n" , hy_val)

'''
0 Cost :  22656.0 
Prediction: 
 [[ 22.04806328]
 [ 21.61978722]
 [ 24.09669304]
 [ 22.29300499]
 [ 18.6339016 ]]
 
 2000 Cost :  3.17888 
Prediction: 
 [[ 154.35929871]
 [ 182.95117188]
 [ 181.8505249 ]
 [ 194.35540771]
 [ 142.03565979]]
 
 20000 Cost :  0.885597 
Prediction: 
 [[ 152.32759094]
 [ 184.25119019]
 [ 181.10905457]
 [ 194.73760986]
 [ 142.96763611]]
 
 200000 Cost :  0.15641 
Prediction: 
 [[ 151.55903625]
 [ 184.61074829]
 [ 180.65498352]
 [ 196.06965637]
 [ 142.04722595]]
'''