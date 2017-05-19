###############################################
## Author: Vahid Mirjalili                   ##
## Dept. of Computer Science & Engineering   ##
## Michigan State University                 ##
###############################################

import tensorflow as tf
import numpy as np

#######################
## Wrapper Functions ##
#######################

def conv_layer(input_tensor, name,
               kernel_size=(3,3), n_filters=256, 
               padding_mode='SAME', strides=(1,1,1,1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape: [batch x width x height x channels_in]
        n_input_channels = input_tensor.get_shape().as_list()[-1] 

        weights_shape = list(kernel_size) + \
                        [n_input_channels, n_filters]

        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape,
                                  initializer=weights_init)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(shape=[n_filters]))
        conv = tf.nn.conv2d(input=input_tensor, filter=weights,
                            strides=strides, padding=padding_mode)
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv)
        print(weights)
        
        return conv

## testing    
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None,28,28,1])
conv_layer(x, name='test', kernel_size=(3,3), n_filters=32)

def pool_layer(input_tensor, name):
    return tf.nn.max_pool(input_tensor, 
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], 
                          padding='SAME', 
                          name=name)

def fc_layer(input_tensor, name, 
             n_output_units, dropout=None, 
             activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, 
                                      shape=(-1, n_input_units))
        if dropout is not None:
            input_tensor = tf.nn.dropout(input_tensor, keep_prob=dropout)

        weights_shape = [n_input_units, n_output_units]
        weights_init = tf.contrib.layers.xavier_initializer()
        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape,
                                  initializer=weights_init)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(shape=[n_output_units]))
        layer = tf.matmul(input_tensor, weights)
        layer = tf.nn.bias_add(layer, biases)
        print(weights)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer)
        return layer
    
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None,28,28,1])
fc_layer(x, name='fc', n_output_units=32, activation_fn=tf.nn.relu)

#############################
###      VGG16 Class      ###
#############################

class VGG16(object):
    def __init__(self, n_features, n_classes,
                 epochs=40, learning_rate=1e-5, dropout=0.5,
                 shuffle=True, random_state = None, weight_file=None, 
                 initialize=False):
        np.random.seed(random_state)
        self.n_features = n_features  # a tuple
        self.n_classes = n_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
       
        g = tf.Graph() 
        with g.as_default():
            ## build the network
            self.build()
            self.saver = tf.train.Saver()

            ## create a session:
            self.sess = tf.Session(graph=g)

            if weight_file is not None:
                self.load_params(weight_file)
            elif initialize:
                ##initialize variables
                try:
                    self.sess.run(tf.global_variables_initializer())
                except:
                    self.sess.run(tf.initialize_all_variables())

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)
 
    def build(self):
        
        self.tf_x = tf.placeholder(tf.float32, 
                                   shape=[None, 
                                          self.n_features[0], 
                                          self.n_features[1], 
                                          self.n_features[2]],
                                   name='inputs')
        self.tf_y = tf.placeholder(tf.int32, shape=[None], 
                                   name='targets')
        tf_y_onehot = tf.one_hot(self.tf_y, depth=self.n_classes)
        print(' ** x : ', self.tf_x)
        print(' ** y_ : ', self.tf_y)
        print(' ** y1hot: ', tf_y_onehot)

        ### Convolutional Layers ###
        h_conv1_1 = conv_layer(self.tf_x, name='conv1_1', n_filters=64)
        h_conv1_2 = conv_layer(h_conv1_1, name='conv1_2', n_filters=64)
        h_pool1   = pool_layer(h_conv1_2, name='pool1')

        h_conv2_1 = conv_layer(h_pool1,   name='conv2_1', n_filters=128)
        h_conv2_2 = conv_layer(h_conv2_1, name='conv2_2', n_filters=128)
        h_pool2   = pool_layer(h_conv2_2, name='pool2')

        h_conv3_1 = conv_layer(h_pool2,   name='conv3_1', n_filters=256)
        h_conv3_2 = conv_layer(h_conv3_1, name='conv3_2', n_filters=256)
        h_conv3_3 = conv_layer(h_conv3_2, name='conv3_3', n_filters=256)
        h_pool3   = pool_layer(h_conv3_3, name='pool3')

        h_conv4_1 = conv_layer(h_pool3,   name='conv4_1', n_filters=512)
        h_conv4_2 = conv_layer(h_conv4_1, name='conv4_2', n_filters=512)
        h_conv4_3 = conv_layer(h_conv4_2, name='conv4_3', n_filters=512)
        h_pool4   = pool_layer(h_conv4_3, name='pool4')

        h_conv5_1 = conv_layer(h_pool4,   name='conv5_1', n_filters=512)
        h_conv5_2 = conv_layer(h_conv5_1, name='conv5_2', n_filters=512)
        h_conv5_3 = conv_layer(h_conv5_2, name='conv5_3', n_filters=512)
        h_pool5   = pool_layer(h_conv5_3, name='pool5')


        ## 1st FC Layer
        self.keep_prob_fc1 = tf.placeholder(tf.float32, name='keep_prob_fc1')
        print(self.keep_prob_fc1)
        h_fc1 = fc_layer(h_pool5, name='fc1', 
                         n_output_units=1024,  ## original: 4096
                         dropout=self.keep_prob_fc1,
                         activation_fn=tf.nn.relu)
        ## 2nd FC Layer
        self.keep_prob_fc2 = tf.placeholder(tf.float32, name='keep_prob_fc2')
        print(self.keep_prob_fc2)
        h_fc2 = fc_layer(h_fc1, name='fc2', 
                         n_output_units=1024, ## original: 4096
                         dropout=self.keep_prob_fc2,
                         activation_fn=tf.nn.relu)
        ## 3rd FC Layer
        logits = fc_layer(h_fc2, name='fc3', 
                         n_output_units=self.n_classes, 
                         dropout=None, activation_fn=None)

        self.predictions = {
            'labels' : tf.cast(tf.argmax(logits, 1), tf.int32),
            'probabilities' : tf.nn.softmax(logits)
        }
        
        ## Loss Function and Optimization
        self.cross_entropy = tf.reduce_mean(
                               tf.nn.softmax_cross_entropy_with_logits(
                                   logits = logits, 
                                   labels = tf_y_onehot),
                               name='cross_entropy_loss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='adam_ptimizer')
        self.optimizer = optimizer.minimize(self.cross_entropy, name='optimizer_minimize_loss')
        
        ## Finding accuracy
        correct_prediction = tf.equal(self.predictions['labels'], 
                                      self.tf_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        
    def train_batch(self, batch_x, batch_y):
        
        _, loss = self.sess.run([self.optimizer, self.cross_entropy], 
                                feed_dict={self.tf_x :batch_x, 
                                           self.tf_y: batch_y, 
                                           self.keep_prob_fc1: self.dropout,
                                           self.keep_prob_fc2: self.dropout})
        return loss
                    
    def predict(self, X, return_proba=False):
        predictions = self.sess.run(self.predictions, 
                                    feed_dict={self.tf_x:X, 
                                               self.keep_prob_fc1: 1.0, 
                                               self.keep_prob_fc2:1.0})
        if return_proba:
            return predictions['probabilities']
        else:
            return predictions['labels']



