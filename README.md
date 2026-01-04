<h1 align="center">
High-Precision Force Feedback Control Method for MR Brakes <br>
Based on an Improved GRU and Bouc–Wen Model
</h1>

<p align="center">
Dapeng Chen, Zhenjie Ma, Haojun Ni, Hong Zeng, Jia Liu, and Aiguo Song<br>
Nanjing University of Information Science and Technology
</p>

<h2 align="center">ABSTRACT</h2>

As a compact, low-power, and safe actuator, Magnetorheological (MR) brake is commonly used to provide passive force feedback for haptic interactions. However, due to the influence of hysteresis effect, temperature drift and environmental disturbances, MR brake is prone to issues with insufficient output force accuracy. To enable the MR actuator to deliver rapid, precise and stable force feedback during force-tactile interactions, we propose a composite force feedback control framework that integrates feedforward modeling with inverse mapping. First, an improved Fractional Asymmetric Bouc-Wen (FABW) forward model was constructed to address the hysteresis effect and rate-dependent characteristics. This forward model accurately characterizes the dynamic nonlinear behavior of the MR brake by incorporating fractional-order memory terms, tanh asymmetric correction terms, and disturbance compensation. Secondly, to address the challenge of balancing accuracy and real-time performance when solving the inverse of a forward model, we propose an inverse model based on an enhanced Gated Recurrent Unit (GRU). This approach achieves low-latency, high-precision mapping from desired forces to excitation currents. Finally, considering that time-varying nonlinearities and drive delays can introduce residuals in the theoretical current of the inverse model, a PID controller combined with a near-end strategy optimization algorithm was introduced. This further enhanced the tracking accuracy and robustness of the force feedback control system. Experimental results demonstrate that the proposed FABW forward model exhibits significantly lower prediction errors than traditional models. The inverse model achieves an RMSE of 0.038 A for current prediction, while the system maintains a force tracking error of merely 0.052 N during virtual grasping tasks. This effectively enhances the realism and immersion of force feedback interaction.


## Method Overview

This project proposes a high-precision force feedback control framework for magnetorheological (MR) brakes by integrating a physics-informed forward model and a data-driven inverse model.

The overall control pipeline consists of three key components:

1. **Forward Model (FABW)**  
   An improved Fractional Asymmetric Bouc–Wen (FABW) model is developed to accurately characterize the hysteresis, rate-dependence, and asymmetry of MR brakes under varying operating conditions.

2. **Inverse Model (VMD-GRU-Attention)**  
   A data-driven inverse model based on Variational Mode Decomposition (VMD), Gated Recurrent Unit (GRU), and Attention mechanism is designed to map the desired force to the required excitation current with low latency. The overall structure of the inverse model is shown below.
<p align="center">
  <img src="fig1.jpg" width="450">
</p>

3. **Closed-loop Compensation (PPO-PID)**  
   A PPO-optimized PID controller compensates modeling errors and disturbances, ensuring robust and accurate force tracking in real-time interaction tasks.The flowchart of the compound control algorithm for the MR brake is shown below.
<p align="center">
  <img src="fig2.jpg" width="500">
</p>

<h3 align="center">IMPLEMENTATION DETAILS</h3>

## Environment Setup

Configure necessary environment packages based on `requirements.txt`.


## Parameter path setting
Enter `configs/configs.py` to adjust the path of the data file.
```bash
configs/configs.py
```


## DATASET
The dataset we used was derived from an experiment platform of a direct-acting magnetorheological (MR) brake independently developed by us. By setting up a complete mechanical test and control system, dynamic response data of the input current and output force of the MR brake under different displacement and speed conditions were collected and used as the training and test sets for the model. The partially collected current and output force data can be viewed in the folder `The dataset we collected`.

## Data Preparation
Start by running `utils/data_deal.py` to retrieve and organize the required original dataset.
```bash
utils/data_deal.py
```
Then run `utils/dataprocess.py`, align the excel data by file name and merge it into a csv file. Crop the data before and after and save it as a standardized result.
```bash
utils/dataprocess.py
```

## Model Training
Adjust "args. is_training==1" in `configs/configs.py`, and use `main.py` to train the model. Adjust parameters such as epoch and batch_2 according to the training environment.
```bash
main.py
```

## Testing
Adjust "args. is_training==2" in `configs/configs.py`, and use `main.py` to validate results on the test set.




