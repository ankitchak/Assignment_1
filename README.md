# Assignment_1

## Tensorflow:

TensorFlow is an open source software library for numerical computation using data flow graphs. The graph nodes represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture enables you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.

We are using tensorflow to get the postion.

## Update Equations:

Updating positions and velocities can be done at infinitesimal time steps ∆t according to the
following equations,

x_t+∆t = x_t + v_t * ∆t + a_t * ∆t^2

v_t+∆t = v_t + a_t * ∆t

where,

x_t+∆t and v t+∆t is the position and velocity of a given particle at time step t + ∆t,
x_t and v_t is the position and velocity of that particle at time step t and
a_t is the acceleration of that particle at time step t.

## Assumptions:

• Assume G = 6.67 × 10 5 units instead of the real G.
• Threshold distance = 0.1 units.
• Time Step (∆t) = 10 −4 units.
