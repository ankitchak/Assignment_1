from typing import Any, Union
import numpy.matlib
import tensorflow as tf
import numpy as np
from numpy import inf
from tensorflow import Tensor


I = tf.eye(100, dtype= tf.float64)
I = I

mass = np.load("/home/souvik/Desktop/DEEP_learning/q2_input/masses.npy")                            # 100 * 1
Initial_position = np.load("/home/souvik/Desktop/DEEP_learning/q2_input/positions.npy")              # 100 * 2
Initial_velocity = np.load("/home/souvik/Desktop/DEEP_learning/q2_input/velocities.npy")              # 100 * 2

mass = tf.convert_to_tensor(mass, dtype=tf.float64)
Initial_position = tf.convert_to_tensor(Initial_position, dtype=tf.float64)
Initial_velocity = tf.convert_to_tensor(Initial_velocity, dtype=tf.float64)

G = tf.dtypes.cast(tf.constant(6.67 * (10 ** 5)), tf.float64)                 # Gravitational constant
Th = tf.dtypes.cast(tf.constant(0.1), tf.float64)                             # Threshold distance
del_t = tf.constant(10 ** (-4), tf.float64)                   # Time Step

#pos = tf.placeholder(tf.float64,None)
#vel = tf.placeholder(tf.float64,None)


def _rel_dist_x_(position):
    p_x = tf.dtypes.cast(tf.reshape(position[:, 0], (100, 1)), tf.float64)
    matrix = tf.dtypes.cast(tf.ones([1, 100]), tf.float64)
    p_mat_x = tf.linalg.matmul(p_x, matrix)
    r_n_x = tf.math.subtract(p_mat_x, tf.transpose(p_mat_x))
    return r_n_x


def _rel_dist_y_(position):
    p_y = tf.dtypes.cast(tf.reshape(position[:, 1], (100, 1)), tf.float64)
    matrix = tf.dtypes.cast(tf.ones([1, 100]), tf.float64)
    p_mat_y = tf.linalg.matmul(p_y, matrix)
    r_n_y = tf.math.subtract(p_mat_y, tf.transpose(p_mat_y))
    return r_n_y


def _rel_dist_cube_(position):
    p_x = tf.reshape(position[:, 0], (100, 1))
    p_y = tf.reshape(position[:, 1], (100, 1))
    matrix = tf.dtypes.cast(tf.ones([1,100]), tf.float64)
    p_mat_x = tf.linalg.matmul(p_x, matrix)
    p_mat_y = tf.linalg.matmul(p_y, matrix)
    r_n_x_2 = tf.math.square(tf.math.subtract(p_mat_x, tf.transpose(p_mat_x)))
    r_n_y_2 = tf.math.square(tf.math.subtract(p_mat_y, tf.transpose(p_mat_y)))
    r_n_2 = tf.math.add(r_n_x_2, r_n_y_2)
    rel_n = tf.math.sqrt(r_n_2)
    rel_d_3 = tf.pow(rel_n, 3)
    rel_d_3 = tf.math.add(rel_d_3,I)
    return rel_d_3

def _acceleration_(position):
    m = tf.dtypes.cast(tf.tile(mass, (1,100)), tf.float64)
    d = tf.math.divide(m, _rel_dist_cube_(position))
    s_x = tf.reshape(tf.math.reduce_sum(tf.math.multiply(d, _rel_dist_x_(position)), 1), (100, 1))
    s_y = tf.reshape(tf.math.reduce_sum(tf.math.multiply(d, _rel_dist_y_(position)), 1), (100, 1))
    a = tf.concat((s_x, s_y), 1)
    acc = tf.math.negative(tf.math.multiply(G, a))
    return acc


def _position_update_(position, velocity, acceleration):
    a_t_2 = tf.math.multiply(tf.math.square(del_t), acceleration)
    half_a_t_2 = tf.math.multiply(tf.dtypes.cast(0.5, tf.float64), a_t_2)
    v_t = tf.math.multiply(velocity, del_t)
    p_t_update = (position+ v_t+ half_a_t_2)
    return p_t_update


def _velocity_update_(velocity, acceleration):
    a_t = tf.math.multiply(acceleration, del_t)
    v_t_update = tf.math.add(velocity, a_t)
    return v_t_update


def _update_(position, velocity, acceleration):
    position = _position_update_(position, velocity, acceleration)
    velocity = _velocity_update_(velocity, acceleration)
    return  position, velocity


def _print_(position, velocity):
    print((position))
    print((velocity))
#print(sess.run(count))

count=0
with tf.Session() as sess:

    pos = Initial_position
    vel = Initial_velocity
    #for i in range(0):
    relative_dist_3 = _rel_dist_cube_(pos)
    r_n = tf.math.pow(relative_dist_3, (1/3))
    #tf.matrix_set_diag(r_n, 0.2)
    accel = _acceleration_(pos)
    for i in range(197):
        pos = _position_update_(pos, vel, accel)
        vel = _velocity_update_(vel, accel)
        count=count+1

        r=tf.where(tf.reduce_min(r_n) <= Th,x=print(sess.run(pos)),y=None)
        #print(sess.run(pos))

