#coding: utf-8

"""
this is the model file of emr disambiguation.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug


"""
Model Class
"""

class Model:
    """ Class that defines a graph to normalize medical concepts with EMR """

    def __init__(self, config):
        """ Creates a model for entity disambiguation

        Args:
            config: include hyper parameters
        """
        self.config = config
        tf.reset_default_graph()
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, self.config['x1_maxlen']))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, self.config['x2_maxlen']))

        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None,))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None,))

        self.Y = tf.placeholder(tf.int32, name=(None,))

        self.embedding = tf.get_variable('embedding', initializer=self.config['embedding'],
                                         dtype=tf.float32, trainable=False)

        self.embed1 = tf.nn.embedding_lookup(self.embedding, self.X1)
        self.embed2 = tf.nn.embedding_lookup(self.embedding, self.X2)


        # local attention layer
        # W matrix: m * m
        self.w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention'], self.config['dimention']))
        self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer(), dtype=tf.float32, shape=self.config['dimention'])

        # U matrix: m * m
        self.u1 = tf.get_variable('u1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention'], self.config['dimention']))
        self.b2 = tf.get_variable('b2', initializer=tf.constant_initializer(), dtype=tf.float32, shape=self.config['dimention'])

        # V matrix: m * 1
        self.v1 = tf.get_variable('v1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention']))

        self.local_attention = tf.placeholder(tf.float32, name='local_attention',
                                              shape=(None, self.config['x2_maxlen'], self.config['dimention']))

        # multi-task shared layer

        self.w2 = tf.get_variable('w2', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention'], self.config['dimention']))
        self.b3 = tf.get_variable('b2', initializer=tf.constant_initializer(), dtype=tf.float32, shape=self.config['dimention'])

        self.u2 = tf.get_variable('u2', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention'], self.config['dimention']))
        self.b4 = tf.get_variable('b2', initializer=tf.constant_initializer(), dtype=tf.float32,
                                  shape=self.config['dimention'])
        self.v2 = tf.get_variable('v2', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(self.config['dimention']))

        self.shared = tf.placeholder(tf.float32, name='multi_task_shared',
                                     shape=(self.config['batch_size'], self.config['dimension']))

        # multi-task mlp input layer

        self.mlp = tf.placeholder(tf.float32, name='multi_task_mlp',
                                  shape=(None, self.config['x2_maxlen'], 2*self.config['dimention']))

        self.mlp_w1 = tf.get_variable('mlp_w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, shape=(2*self.config['dimention'], self.config['mlp_layer_2']))
        self.mlp_b1 = tf.get_variable('mlp_b1', initializer=tf.constant_initializer(), dtype=tf.float32, shape=self.config['mlp_layer_2'])

        #self.fc1 = tf.placeholder(tf.float32, name='mlp_fc1', shape=(None, self.config['mlp_layer_2']))
        self.fc1 = self.activation_layer(self.mlp, self.mlp_w1, self.mlp_b1)

        self.mlp_w2 = tf.get_variable('mlp_w2', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                                                        dtype=tf.float32),
                                      dtype=tf.float32, shape=(self.config['mlp_layer_2'], self.config['entity_class_num']))
        self.mlp_b2 = tf.get_variable('mlp_b2', initializer=tf.constant_initializer(), dtype=tf.float32,
                                      shape=self.config['entity_class_num'])

        #self.fc2 = tf.placeholder(tf.float32, name='mlp_fc2', shape=(None, self.config['entity_class_num']))
        self.fc2 = self.activation_layer(self.fc1, self.mlp_w2, self.mlp_b2)

        # softmax output

        self.pred = tf.nn.softmax(self.fc2)

        # cross entropy as loss function

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.pred, 1e-10)),
                                                  reduction_indices=[1]))
        self.predictions, self.actuals = tf.arg_max(self.pred, 1), tf.arg_max(self.Y, 1)

        # train model with SGD
        self.train_model = tf.train.AdadeltaOptimizer().minimize(self.loss)


    def init_step(self, sess):
        """ initializer all variables """

        sess.run(tf.global_variables_initializer())


    def cal_accu(self):
        """ calculate accuracy, precision, recall, f1 """

        ones_like_actuals = tf.ones_like(self.actuals)
        zeros_like_actuals = tf.zeros_like(self.actuals)
        ones_like_predictions = tf.ones_like(self.predictions)
        zeros_like_predictions = tf.zeros_like(self.predictions)

        TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, ones_like_actuals),
                                                       tf.equal(self.predictions, ones_like_predictions, 'float'))))
        TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, zeros_like_actuals),
                                                       tf.equal(self.predictions, zeros_like_predictions)), 'float'))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, zeros_like_actuals),
                                                       tf.equal(self.predictions, ones_like_predictions)), 'float'))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, ones_like_actuals),
                                                       tf.equal(self.predictions, zeros_like_predictions)), 'float'))

        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        accu = (TP + TN) * 1.0 / (TP + TN + FP + FN)

        return precision, recall, f1, accu



    def extend_matrix_to_tensor(self, multiple_num, flag):
        """ extend a matrix of batch_size * m * p to batch_size * m * p * q """

        if flag == 'f_x1':
            s_tensor = tf.einsum('ii, aij->aij', self.w1, self.X1) + self.b1
        else:
            s_tensor = tf.einsum('ii, aij->aij', self.u1, self.X2) + self.b2
        s_extend_tensor = tf.tile(s_tensor, multiple_num)
        s_extend_tensor = tf.reshape(s_extend_tensor,
                                     [self.config['batch_size'], self.config['x2_maxlen'],
                                      self.config['dimention'], self.config['x1_maxlen']])

        if flag == 'f_x1':
            s_extend_tensor = tf.transpose(s_extend_tensor, [0, 2, 3, 1])
        else:
            s_extend_tensor = tf.transpose(s_extend_tensor, [0, 2, 1, 3])
        return s_extend_tensor


    def get_local_attention(self):
        """ calculate local attention layer

            equations:
                Ci = sum(ai * xj)
                ai = softmax(eij)
                eik = V*tanh(WXi + UXj)
        """

        # transpose batch_size * p * m to batch_size * m * p
        self.X1 = tf.transpose(self.X1, [0, 2, 1])
        self.X2 = tf.transpose(self.X2, [0, 2, 1])

        s_tensor = self.extend_matrix_to_tensor([self.config['x2_maxlen'], 1], 'f_x1')
        d_tensor = self.extend_matrix_to_tensor([self.config['x1_maxlen'], 1], 'f_x2')

        # batch_size * m * p * q
        f_tensor = tf.tanh(s_tensor + d_tensor)

        # get tensor slice: batch_size * m * p
        #f_tensor = tf.slice(f_tensor, [0, 0, 0, 0], [-1, 1, self.config['x1_maxlen'], 1])

        # bath_size * p * q
        e_tensor = tf.einsum('ii, aijl->ajl', self.v1, f_tensor)


        one_hot_tensor = self.dense2one_hot()
        alphas = self.my_softmax(e_tensor, one_hot_tensor)

        # batch_size * q * m
        output = tf.einsum('aji,il->ajl', self.X1, alphas)
        return output


    def dense2one_hot(self):
        """ generate special one_hot tensor """
        num_labels = self.config['x1_maxlen']
        pass





    def my_softmax(self, e_tensor, one_hot_tensor):
        """ calculate softmax by myself """
        exp_tensor = tf.exp(e_tensor)
        pass


    def extend_matrix_to_tensor2(self, w_tensor, u_tensor):
        """ extend a matrix to tensor from batch_size * m * q to batch_size * m * q * q """
        w_tensor = tf.einsum('ii, aij->aij', self.w2, w_tensor) + self.b3
        u_tensor = tf.einsum('ii, aij->aij', self.u2, u_tensor) + self.b4

        w_extend_tensor = tf.tile(w_tensor, [self.config['x2_maxlen'], 1])
        u_extend_tensor = tf.tile(u_tensor, [self.config['x2_maxlen'], 1])

        w_extend_tensor = tf.reshape(w_extend_tensor,
                                     [self.config['batch_size'], self.config['dimention'],
                                      self.config['x2_maxlen'], self.config['x2_maxlen']])
        u_extend_tensor = tf.reshape(u_extend_tensor,
                                     [self.config['batch_size'], self.config['dimention'],
                                      self.config['x2_maxlen'], self.config['x2_maxlen']])
        return w_extend_tensor, u_extend_tensor

    def shared_layer(self):
        """ calculate attention shared layer information
            Equations:
                Share = sum(bi * ci)
                bi = softmax(ei)
                ei = sum_j(g(ci, dj))
                g(ci, dj) = V*tanh(WC + UD)

            input: batch_size * q * m
            output: batch_size * m
        """
        w_tensor = tf.transpose(self.local_attention, [0, 2, 1])
        u_tensor = self.X2

        w_tensor, u_tensor = self.extend_matrix_to_tensor2(w_tensor, u_tensor)

        # batch_size * m * q * q
        g_tensor = tf.tanh(w_tensor + u_tensor)

        # batch_size * q * q
        g_tensor = tf.einsum('i, aijj->ajj', tf.transpose(self.v2), g_tensor)


        g_diag_tensor = tf.diag(g_tensor, 1)

        # batch_size * q
        e_tensor = tf.reduce_sum(g_tensor) - tf.reduce_sum(g_diag_tensor)

        one_hot_tensor = self.dense2one_hot()
        betas = self.my_softmax(e_tensor, one_hot_tensor)

        # batch_size * q * m
        output = tf.einsum('aji,ail->ajl', self.local_attention, betas)
        return output


    def get_mlp_input(self):
        """ merge shared info with each task to form mlp input
            input:
                shared info: batch_size * q * m
                task info: batch_size * q * m
            output: batch_size * q * 2m
        """
        return tf.concat(3, [self.local_attention, self.shared])


    def activation_layer(self, fc, w, b):
        """ get activation layer """
        return tf.nn.relu(tf.matmul(fc, w) + b)


    def train_step(self, sess, feed_dict):
        """ train model """
        feed_dict[self.local_attention] = self.get_local_attention()


    def test_step(self, sess, feed_dict):
        """ test model """
        pass
