Physics-Informed Neural Networks (PINNs) Project
This repository contains implementations of Physics-Informed Neural Networks (PINNs) for solving various differential equations and related tasks. The main objective is to solve Ordinary Differential Equations (ODEs) by training neural networks that incorporate physical laws (via differential equations) into the learning process. This approach allows the model to fit the data and estimate unknown parameters or functions. Additionally, the project includes a simple implementation of a Physics-Informed Deep Operator Network (DeepONet).

Project Structure
The project consists of four Python scripts:

1. cooling.py
This script implements a simple PINN to solve a cooling problem. The goals include:

Fitting data generated from a cooling process.
Estimating one of the parameters within the governing equation (e.g., a thermal constant).
This example demonstrates how a PINN can approximate the solution and infer hidden parameters.

2. oscillator_period.py
This script solves an ODE describing an oscillator's motion, focusing on determining the oscillation period. The challenges include:

Solving a second-order ODE.
Estimating the oscillation period.
This task is more complex than cooling.py due to the second-order nature of the ODE and the indirect target.

3. Kraichnan-Orszag.py
This script implements a PINN to solve the Kraichnan-Orszag system, a set of three interrelated ODEs describing a dynamic system. The main objectives are:

Learning the structure of the equations connecting three parameters.
Estimating two hyperparameters in the equations.
This advanced example involves learning both the functional relationships and system dynamics.

4. PI_DeepONet.py
This script implements a simple Physics-Informed DeepONet. Key features include:

Using a DeepONet to learn mappings between function spaces.
Adding losses like the ODE loss and initial condition (IC) loss to improve learning.
This showcases the extension of PINNs to DeepONets for more flexible system approximations.


