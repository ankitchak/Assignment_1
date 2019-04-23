from typing import Any, Union
import numpy.matlib
import tensorflow as tf
import numpy as np
from numpy import inf
from tensorflow import Tensor


I = tf.eye(100, dtype= tf.float64)
I = I
output="/home/supriyo/practice/"

mass = np.load("/home/supriyo/practice/q2_input/masses.npy")                            # 100 * 1
position = np.load("/home/supriyo/practice/q2_input/positions.npy")              # 100 * 2
velocity = np.load("/home/supriyo/practice/q2_input/velocities.npy")              # 100 * 2


#mass=tf.constant(mass,tf.float64)
#mass = tf.convert_to_tensor(mass, dtype=tf.float64)
#Initial_position = tf.convert_to_tensor(Initial_position, dtype=tf.float64)
#Initial_velocity = tf.convert_to_tensor(Initial_velocity, dtype=tf.float64)

G = tf.dtypes.cast(tf.constant(6.67 * (10 ** 5)), tf.float64)                 # Gravitational constant
Th = 0.1                             # Threshold distance
del_t = tf.constant(10 ** (-4), tf.float64)                   # Time Step

#pos = tf.placeholder(tf.float64,None)
#vel = tf.placeholder(tf.float64,None)

n=mass.shape[0]


#various place holder that will take values later on
x=tf.placeholder(tf.float64,[n,2],name="x")
m=tf.placeholder(tf.float64,[n,1],name="m")
v=tf.placeholder(tf.float64,[n,2],name="v")




#this function check that a pair of object exits having distance betwen them is less than threshold    
def check_pair(x,Th):
    n=x.shape[0]
    for i in range(n):
        for j in range(n):
            if i!=j:
                dist=pow(pow(x[j][0]-x[i][0],2)+pow(x[j][1]-x[i][1],2),0.5)
                if dist<= Th:
                    return 1,i,j
    return 0,i,j


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

a=_acceleration_(x)
v_final=_velocity_update_(v, a)
x_final=_position_update_(x, v, a)



#def _update_(position, velocity, acceleration):
#    position = _position_update_(position, velocity, acceleration)
#    velocity = _velocity_update_(velocity, acceleration)
#    return  position, velocity


#def _print_(position, velocity):
#    print((position))
#    print((velocity))
#print(sess.run(count))



#count=0
#here we start the session to compute the final velocities and positions of particles
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output_log", sess.graph)
    a1=0;j=0
    while(a1!=1):
        fd={x:position,v:velocity,m:mass}
        v_f=sess.run(v_final,feed_dict=fd)
        x_f=sess.run(x_final,feed_dict=fd)

        velocity=v_f
        position=x_f
        a1,b1,c1=check_pair(position,Th)
        j+=1
        if a1==1:
            #print(j,b1,c1)
            np.save(output+"positions.npy",position)
	    #print(sess.run('final_position'+str(x1)))
            np.save(output+"velocities.npy",velocity)
            #print(sess.run('final_velocity'+str(v1)))
            break;
    # for i in range(n):
    #     print(i,x1[i])
    # print(v_f)
    writer.close()

print("In",j,"th iteration distance between ",b1,"th and ",c1,"th particle becomes less than ",Th)
