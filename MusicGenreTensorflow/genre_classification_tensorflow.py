import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils import np_utils
from tensorflow.contrib.data import Dataset, Iterator

def load_data():
    x_train = np.load( "features-train.npy" )
    y_train = np.load( "classes-train.npy" )
    x_test = np.load( "features-test.npy" )
    y_test = np.load( "classes-test.npy" )
    return x_train,y_train,x_test, y_test

def RNN(x):
    x = tf.unstack( x, time_steps, 1)

    lstm_cell = rnn.BasicLSTMCell( num_hidden, forget_bias=1.0)

    # get lstm output
    outputs, hidden_states = rnn.static_rnn( lstm_cell, x, dtype=tf.float32)

    w = tf.Variable( tf.random_normal([num_hidden, num_classes]))
    b = tf.Variable( tf.random_normal([num_classes]))
    return tf.matmul( outputs[-1],w) + b


if __name__ == "__main__":
    time_steps = 128
    num_hidden = 128
    num_classes = 2
    feature_size = 33
    learning_rate = 0.001
    training_steps = 100#00
    display_step = 100
    x_train,y_train,x_test, y_test = load_data()
    y_train = np_utils.to_categorical( y_train )
    y_test = np_utils.to_categorical( y_test )

    x = tf.placeholder( "float", [None, time_steps, feature_size])
    y = tf.placeholder( "float", [None,num_classes])


    logits = RNN( x )
    prediction = tf.nn.softmax( logits )

    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y ) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize( loss )

    correct_pred = tf.equal( tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean( tf.cast(correct_pred,tf.float32))
    init = tf.global_variables_initializer()

    # create tensorflow dataset objects
    train_data = Dataset.from_tensor_slices((x_train, y_train))
    test_data = Dataset.from_tensor_slices((x_test, y_test))



    #training_init_op = iterator.make_initializer(train_data)


    with tf.Session() as sess:
        sess.run( init )
        for step in range(1, training_steps+1):
            losses = []
            acc_avg = []
            iterator = Iterator.from_structure( train_data.output_types, train_data.output_shapes)
            next_element = iterator.get_next()
            training_init_op = iterator.make_initializer(train_data)
            sess.run( training_init_op )
            print("Step: ", step)
            while True:
                try:
                    x_element, y_element = sess.run( next_element )
                    x_element = x_element.reshape((1,time_steps, feature_size))
                    y_element = y_element.reshape((-1,2))
                    sess.run( [loss, accuracy,train], feed_dict={x:x_element, y:y_element})
                    #loss, acc = sess.run( [loss, accuracy], feed_dict={x:x_element, y:y_element})
                    loss_value, acc = sess.run( [loss, accuracy], feed_dict={x:x_element, y:y_element})
                    losses.append( loss_value )
                except tf.errors.OutOfRangeError:
                    #print("End of training set")
                    break
            #if step % display_step == 0  or step == 1:
            loss_sum = np.sum( np.array( losses ) )
            loss_avg = loss_sum/ len(losses)
            print("Loss Average: ", loss_avg)
                #loss, acc = sess.run( [loss, accuracy], feed_dict={x:x_element, y:y_element})
                #print( "Loss: ", loss)
                #print(" Accuracy: ", acc)
        x_test = x_test.reshape((-1,time_steps, feature_size))
        y_test = y_test.reshape((-1,2))
        print( "Test Accuracy: ", sess.run( accuracy, feed_dict={x:x_test, y:y_test}))
