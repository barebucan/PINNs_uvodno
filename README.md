Physics-Informed Neural Networks (PINNs) Project
This repository contains a set of implementations of Physics-Informed Neural Networks (PINNs) for solving various differential equations and related tasks. The primary goal of this project is to solve Ordinary Differential Equations (ODEs) by training neural networks that incorporate the physical laws (via differential equations) into the learning process. This approach allows the model to not only fit the data but also estimate unknown parameters or functions. The project also includes an implementation of a Physics-Informed Deep Operator Network (DeepONet).

Project Structure
The project consists of four main files:

1. cooling.py
This script implements a simple PINN to solve a cooling problem. The goal is to:

Fit the data generated from the cooling process.
Estimate one of the parameters within the governing equation (for example, a thermal constant in the cooling equation).
This example showcases the use of a PINN to approximate both the solution and a hidden parameter from the data.

2. oscillator_period.py
In this example, we solve an ODE describing the motion of an oscillator, focusing on determining the period of oscillation. The key challenges include:

Solving a second-order ODE, which adds complexity to the PINN training.
Estimating the period of oscillation based on the model's predictions.
This problem is more challenging than the cooling.py example because it involves a second-order differential equation and an indirect target (oscillation period).

3. Kraichnan-Orszag.py
This script implements a PINN model to solve the Kraichnan-Orszag system, which consists of three interrelated ODEs governing the dynamics of a physical system. The key tasks include:

Learning the structure of the equations connecting the three parameters.
Estimating two hyperparameters within the functions.
In this more advanced example, the model learns both the functional relationships between the parameters and the underlying system dynamics.

4. PI_DeepONet.py
This script implements a Physics-Informed DeepONet, which is a neural operator designed to handle complex systems of equations. Key features include:

The DeepONet architecture, which can learn mappings between function spaces.
Added losses including the ODE loss and initial condition (IC) loss to guide the learning process.
This example demonstrates how to extend PINNs to DeepONets, allowing for more flexible and powerful approximations of complex systems.

