import tensorflow as tf

hello=tf.constant("hello , Tensorflow")

sess=tf.Session()
print(sess.run(hello))

node1 = tf.constant(3.0 , tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(4.0)
node4 = tf.add(node1 , node2)
node5 = tf.add(node4 , node3)

print(node1 , node2 , node3  , node4)
print(sess.run( [node1 , node2 ]))
print(sess.run( node4))
print(sess.run(node5))
# add는 2개의 파라미터만 가능하다... 3개 add 는 안된다. 단계를 거쳐서 계산해야가능하다.

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node = a + b #tf.add(node4 , node3) 이것의 shortcut 이다. ㅇㅋ
print(sess.run(adder_node , feed_dict={a:3 , b: 4.5})) # placeholder 한번에 값 줄때 {} 사용한다.
print(sess.run(adder_node , feed_dict={a:[1,3] , b:[2,4] } ) )  # 각 placeholder 에 여러개의 값을 줄때는 []을 사용한다.