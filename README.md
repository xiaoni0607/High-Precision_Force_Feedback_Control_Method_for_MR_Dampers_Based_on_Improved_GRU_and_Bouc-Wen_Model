# High-Precision Force Feedback Control Method for MR Dampers  Based on an Improved GRU and Bouc–Wen Model

<p align="center">
Your Name, Co-author Name<br>
Nanjing University of Information Science and Technology
</p>


## ABSTRACT

As a compact, low-power, and safe actuator, Magnetorheological (MR) brake is commonly used to provide passive force feedback for haptic interactions. However, due to the influence of hysteresis effect, temperature drift and environmental disturbances, MR brake is prone to issues with insufficient output force accuracy. To enable the MR actuator to deliver rapid, precise and stable force feedback during force-tactile interactions, we propose a composite force feedback control framework that integrates feedforward modeling with inverse mapping. First, an improved Fractional Asymmetric Bouc-Wen (FABW) forward model was constructed to address the hysteresis effect and rate-dependent characteristics. This forward model accurately characterizes the dynamic nonlinear behavior of the MR brake by incorporating fractional-order memory terms, tanh asymmetric correction terms, and disturbance compensation. Secondly, to address the challenge of balancing accuracy and real-time performance when solving the inverse of a forward model, we propose an inverse model based on an enhanced Gated Recurrent Unit (GRU). This approach achieves low-latency, high-precision mapping from desired forces to excitation currents. Finally, considering that time-varying nonlinearities and drive delays can introduce residuals in the theoretical current of the inverse model, a PID controller combined with a near-end strategy optimization algorithm was introduced. This further enhanced the tracking accuracy and robustness of the force feedback control system. Experimental results demonstrate that the proposed FABW forward model exhibits significantly lower prediction errors than traditional models. The inverse model achieves an RMSE of 0.038 A for current prediction, while the system maintains a force tracking error of merely 0.052 N during virtual grasping tasks. This effectively enhances the realism and immersion of force feedback interaction.



## Problem Formulation

The objective of this study is to establish an inverse mapping from the desired output force to the control current of an MR damper, which can be expressed as:
i = g(F_d, x, v)
where:
- `F_d` denotes the desired output force,
- `x` and `v` represent the displacement and velocity states,
- `i` is the input current,
- `g(·)` is the learned inverse control model.

The overall framework integrates a forward hysteresis model and an inverse learning-based controller.

---

## Framework Overview

The proposed force feedback control framework consists of two main components:

1. **Forward Model (Improved Bouc–Wen / FABW):**  
   Used to describe the hysteresis and nonlinear dynamics of the MR damper for analysis and validation.

2. **Inverse Model (Improved GRU):**  
   Learns the inverse mapping from desired force to control current, enabling real-time force feedback control.
