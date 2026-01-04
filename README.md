<h1 align="center">
High-Precision Force Feedback Control Method for MR Dampers <br>
Based on an Improved GRU and Bouc–Wen Model
</h1>

<p align="center">
Your Name, Co-author Name<br>
Nanjing University of Information Science and Technology
</p>

<h2 align="center">ABSTRACT</h2>

As a compact, low-power, and safe actuator, Magnetorheological (MR) brake is commonly used to provide passive force feedback for haptic interactions. However, due to the influence of hysteresis effect, temperature drift and environmental disturbances, MR brake is prone to issues with insufficient output force accuracy. To enable the MR actuator to deliver rapid, precise and stable force feedback during force-tactile interactions, we propose a composite force feedback control framework that integrates feedforward modeling with inverse mapping. First, an improved Fractional Asymmetric Bouc-Wen (FABW) forward model was constructed to address the hysteresis effect and rate-dependent characteristics. This forward model accurately characterizes the dynamic nonlinear behavior of the MR brake by incorporating fractional-order memory terms, tanh asymmetric correction terms, and disturbance compensation. Secondly, to address the challenge of balancing accuracy and real-time performance when solving the inverse of a forward model, we propose an inverse model based on an enhanced Gated Recurrent Unit (GRU). This approach achieves low-latency, high-precision mapping from desired forces to excitation currents. Finally, considering that time-varying nonlinearities and drive delays can introduce residuals in the theoretical current of the inverse model, a PID controller combined with a near-end strategy optimization algorithm was introduced. This further enhanced the tracking accuracy and robustness of the force feedback control system. Experimental results demonstrate that the proposed FABW forward model exhibits significantly lower prediction errors than traditional models. The inverse model achieves an RMSE of 0.038 A for current prediction, while the system maintains a force tracking error of merely 0.052 N during virtual grasping tasks. This effectively enhances the realism and immersion of force feedback interaction.


## Method Overview

This project proposes a high-precision force feedback control framework for magnetorheological (MR) dampers by integrating a physics-informed forward model and a data-driven inverse model.

The overall control pipeline consists of three key components:

1. **Forward Model (FABW)**  
   An improved Fractional Asymmetric Bouc–Wen (FABW) model is developed to accurately characterize the hysteresis, rate-dependence, and asymmetry of MR dampers under varying operating conditions.

2. **Inverse Model (VMD-GRU-Attention)**  
   A data-driven inverse model based on Variational Mode Decomposition (VMD), Gated Recurrent Unit (GRU), and Attention mechanism is designed to map the desired force to the required excitation current with low latency.

3. **Closed-loop Compensation (PPO-PID)**  
   A PPO-optimized PID controller compensates modeling errors and disturbances, ensuring robust and accurate force tracking in real-time interaction tasks.

<h3 align="center">ABSTRACT</h3>

## Environment Setup

Configure necessary environment packages based on `requirements.txt`.
