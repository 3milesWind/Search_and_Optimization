# This is a Python library for Search and Optimization.

This is a library for search and optimization algorithms. The basic topics are covered which include Descent Method, Stochastic Search, Path Search, MDP-related and RL related algorithms. By using this library, you are expected to see basic ideas behind the algorithms through simple but

# Table of Contents

- [What is this?](#What-is-this)
- [Documentation](#Documentation)
- [Numerical Optimization](#Numerical-Optimization)
  - [Gradient Descent](###Gradient-Descent)
  - [Newton Method](###Newton-Method)
  - [Conjugate Descent](###Conjugate Descent)
- [Stochastic Search](#Stochastic-Search)
  - [Simulated Annealing](###Simulated-Annealing)
  - [Cross Entropy Methods](###Cross Entropy Methods)
  - [Search Gradient](###Search Gradient)
- [Classic Search](#Classic-Search)
  - [Depth-First Search](###Depth-First Search)
  - [Breadth-first search (BFS)](###Breadth-first search (BFS))
  - [Dijkstra](###Dijkstra)
  - [A*](###A*)

# Documentation

1. Clone this repo.
   
   git clone [GitHub - 3milesWind/AiRobotics: Algorithms](https://github.com/3milesWind/AiRobotics.git)

# Numerical Optimization

*Numerical Optimization* presents a comprehensive and up-to-date description of the most effective methods in continuous optimization.

### Gradient Descent

Gradient Descent is widely used in optimization and machine learning areas because its simple calculation. It tries to take a small step towards the gradient descent direction to minimize the function.

Example Function: 

- With a fixed learning rate [Code](NumericalOptimization/gradientDescentWithFixedRate.py)
  
  <img src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/NumericalOptimization/gits/GradientDecWithFixedRate.gif" title="" alt="GradientDecWithFixedRate.gif" width="439">

- With a Optimal learning rate [Code](NumericalOptimization/gradientDescentWithOptimalRate.py)

    <img src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/NumericalOptimization/gits/gradient_Decent_Optiomal.gif" title="" alt="gradient_Decent_Optiomal.gif" width="460">

### Newton Method

Example Function: [Code](NumericalOptimization/NewtonMethod.py)

<img src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/NumericalOptimization/gits/newtwonMethod.gif" title="" alt="newtwonMethod.gif" width="509">

Newton's method is a powerful technique—in general the [convergence](https://en.wikipedia.org/wiki/Rate_of_convergence) is quadratic: as the method converges on the root, the difference between the root

However fast and do not need a step size (learning rate), Newton's Method has some drawbacks and caveats:

- The computation cost of inverting the Hessian could be non-trivial.

- It does not work if the Hessian is not invertible.

- It may not converge at all, but can enter a cycle having more than 1 point.

- It can converge to a saddle point instead of to a local minimum

Reference:

See details on [Newton's method in optimization - Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)

### Conjugate Descent

For quadratic problems, we can do better than generic directions of gradient. In Gradient Descent if we take the steepest descent, we always go onthogonal in every step. Can we go faster? Yes, Newton's method gives us a faster one. But at the same time, we also want to avoid calculating the inversion of certain matrics. Conjugate gradients give us a better way to perform descent method because they allow us to minimize convex quadratic objectives in at most n steps and without inverting the matrics.

<img title="" src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/NumericalOptimization/gits/CD_1.gif" alt="CD_1.gif" width="429">

<img title="" src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/NumericalOptimization/gits/CD_2.gif" alt="CD_2.gif" width="427">

# Stochastic Search

In some cases, we do not want to or cannot calculate the first or second derivate of the function. Or there are numerous local minimas of the function and the descent methods not work well. Instead, we can introduce some randomness into the optimization. Here we will cover algorithms including Simulated Annealing, Cross Entropy Method and Search Gradient.

## Simulated-Annealing

Intuitively, Simulated Annealing is to start from one point, dive into the function and walk randomly, go downhill when we can but sometimes uphill to explore and try to jump out of this sub-optimal local minimum. We expect to gradually settle down by reducing the probability of exploring.

For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time, simulated annealing may be preferable to exact algorithms like descent methods.

Here is the visualization of Simulated Annealing:

The Travelling Salesman Problem [Link](TravellingSalesman/Simulated_Annealing.py)

> Simulated annealing algorithm is a random algorithm, it has a certain probability to find the global optimal solution

# Cross Entropy Methods

Instead to take one sample of the function, Cross Entropy Methods sample a distribution. The key idea behind that is that finding a global minimum is equivalent to sampling a distribution centered around it.

Cross Entropy Methods first start with an initial distribution (often a diagonal Gaussian), and then select a subset of samples with lower function values as elite samples. Then update the distribution to best fit those elite samples.

Here is the visualization of CEM, where red points are elite samples.

# Search Gradient

In high dimensions, it can quickly become very inefficient to randomly sample. Ideally, we can use the derivative of the expectation of function value on the distribution we sampled, so that we can move the distribution in the direction that imroves the expectation. So Search Gradient borrows the idea of Gradient Method to do stochastic search. The overall algorithm uses this idea combined with log techniques, see reference for details.

Here is the visualization of Search Gradient:

# Classic-Search

### Depth-First Search (DFS)

The algorithm starts at the root node (selecting some arbitrary node as the root node in the case of a graph) and explores as far as possible along each branch before backtracking.

<img title="" src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/Shortest-Path-of-Maze-Problem/gits/ezgif.com-gif-maker.gif" alt="ezgif.com-gif-maker.gif" width="428">

### Breadth-first search (BFS)

It starts at the tree root and explores all nodes at the present depth prior to moving on to the nodes at the next depth level. Extra memory, usually a queue is needed to keep track of the child nodes that were encountered but not yet explored.



<img title="" src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/Shortest-Path-of-Maze-Problem/gits/ezgif.com-gif-maker%20(1).gif" alt="ezgif.com-gif-maker (1).gif" width="442">

### Dijkstra

For a given source node in the graph, the algorithm finds the shortest path between that node and every other with cost

<img src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/Shortest-Path-of-Maze-Problem/gits/ezgif.com-gif-maker%20(3).gif" title="" alt="ezgif.com-gif-maker (3).gif" width="433">

### A*

A* is an informed search algorithm, **or a best-first search**, meaning that it is formulated in terms of weighted graphs: starting from a specific starting node of a graph, it aims to find a path to the given goal node having the smallest cost

<img src="file:///Users/guoyili/Documents/GitHub/Seach_and_Optimization/Shortest-Path-of-Maze-Problem/gits/ezgif.com-gif-maker%20(4).gif" title="" alt="ezgif.com-gif-maker (4).gif" width="441">
