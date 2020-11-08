# _*_encoding:utf-8_*_
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, batch_norm


class DRNN(object):
    """
    disconnect recurrent neural networks for text categorization
    """
    def __init__(self, sequence_length, num_classes, embedding_size, hidden_size,
                 num_k, batch_size):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_weight = tf.placeholder(tf.int32, [None, sequence_length], name="input_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

#       embedding layer
        #with tf.device('/gpu:0'), tf.name_scope('embedding'):
        with tf.name_scope('embedding'):
            self.embedding_chars = self.input_x

        with tf.name_scope('dropout'):
            self.embedding_chars = tf.nn.dropout(self.embedding_chars, self.dropout_keep_prob)

        with tf.name_scope('embedding_weight'):
            self.W_Weight = tf.Variable(tf.random_uniform([7, 300], -1.0, 1.0, name="W_Weight"))
            self.embedding_placeholder_w = tf.placeholder(tf.float32, [7, 300], name='embedding_placeholder_w')
            self.embedding_init_w = self.W_Weight.assign(self.embedding_placeholder_w, name='embedding_placeholder_w')
            self.embedding_weight = tf.nn.embedding_lookup(self.W_Weight, self.input_weight, name='embedding_weight')

        with tf.name_scope("DGRU_MLP"):
            self.hidden = []
            #print(self.embedding_chars_dropout.shape)
            self.input = tf.pad(self.embedding_chars, [[0, 0], [num_k-1, 0], [0, 0]])
            print(self.input.shape)
            start = 0
            end = start + num_k - 1
            while end < (sequence_length+num_k-1):
                input_k = self.input[:, start:end, :]
                with tf.name_scope("gru"), tf.variable_scope('rnn') as scope:
                    cell = tf.contrib.rnn.GRUCell(hidden_size)
                    # apply dropout in hidden of rnn
                    # cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob)
                    if start != 0:
                        scope.reuse_variables()
                    enconder_outputs, state = tf.nn.dynamic_rnn(cell, input_k, dtype=tf.float32)
                with tf.name_scope("dropout"):
                    state_dropout = tf.nn.dropout(state, self.dropout_keep_prob)
                with tf.name_scope("mlp"), tf.variable_scope('mlp') as scope:
                    if start != 0:
                        scope.reuse_variables()
                    # batch_norm
                    # bn_params = {
                    #     "is_training": self.is_training,
                    #     'decay': 0.99,
                    #     'updates_collections': None
                    # }
                    # mlp_output = fully_connected(state_dropout, 200, scope='mlp', normalizer_fn=batch_norm,
                    #                              normalizer_params=bn_params)
                    W = tf.get_variable("W", shape=[hidden_size, hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
                    mlp_output = tf.nn.relu(tf.nn.xw_plus_b(state_dropout, W, b), name='output')
                    self.hidden.append(mlp_output)
                self.hidden_concat = tf.concat(self.hidden, 1)
                start += 1
                end += 1


        with tf.name_scope("max-pooling"):
            hidden_reshape = tf.reshape(self.hidden_concat, [-1, sequence_length, hidden_size])
            hidden_reshape_expand = tf.expand_dims(hidden_reshape, -1)
            pooled = tf.nn.max_pool(hidden_reshape_expand,
                                    ksize=[1, sequence_length, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='pool')
            pooled_reshape = tf.reshape(pooled, [-1, hidden_size])


        with tf.name_scope("output"):
            #hidden_reshape = tf.reshape(self.hidden_concat, [-1, sequence_length, 200])
            #pooled_reshape = tf.reduce_sum(hidden_reshape, axis=1)
            print('pooled', pooled_reshape)
            self.W2 = tf.get_variable("W2", shape=[hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            self.scores = tf.nn.relu(tf.nn.xw_plus_b(pooled_reshape, self.W2, b2))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #self.scores2 = tf.nn.softmax(self.scores, axis=1)
            self.scores2 = self.scores

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #losses = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=self.scores)
            #self.loss = losses
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


