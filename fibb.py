"""
Tensorflow describes the computation using a static graph. You first have to
define the graph and then execute it.
The graph execution start from the node you put into the
sess.run([var1,var2, ..., vaN]) call: the order of the variables is meaningless.
Tensorflow graph evaluation starts from a random node and follows each node from
the leaf to the root.
Since you want to force a certain order of execution, you have to
use  tf.control_dependencies to introduce an ordering constraint into the
graph and thus for the execution of the operations in a certain order.
"""

import tensorflow as tf

# define the first 2 terms of the Fibonacci sequence
first = tf.Variable(0)
second = tf.Variable(1)

# in TensorFlow we have to force the order of assigments:
# first execute new_term, then execute update1 then update2

# define the next term. By definition it is the sum of the previous ones
new_term = tf.add(first,second)     # alternatively new_term = first + second

with tf.control_dependencies([new_term]):
    update1 = tf.assign(first, second)

    # then execute update2 after update1 and new_term have been executed
    with tf.control_dependencies([update1]):
        update2 = tf.assign(second, new_term)

# initialize all variables
init = tf.global_variables_initializer()

# create graph and run the graph
with tf.Session() as sess:
    sess.run(init)
    fibonacci = [first.eval(), second.eval()]
    for i in range(5):
         next, upd1, upd2 = sess.run([new_term, update1, update2])
         fibonacci.append(next)
    print("The fibonacci sequence is", fibonacci)
