# Unified Sampling Framework (USF)
This is the repository of the paper "A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models".


![framework](<images/USF framework.png>)

we propose a new sampling framework based on the exponential integral formulation that allows free choices of solver strategy at each step and design specific decisions for the framework.

As shown above, our decision space contains 7 element:
1. Discretization timesteps: We allow flexile choice within $[0,T]$.
2. Prediction types: Different formulation of the ODE in different timesteps.
3. Expansion order: The taylor expansion order when updating $x_t$ from $x_s$.
4. Derivative estimation: We allow different estimation order for each derivatives, and choose some flexible relaxation coefficients on the derivatives.
5. Corrector: Whether to use pseudo corrector or not. Corrector was proved to be useful on some timesteps, but not all of them.
6. Analytical first step: With ground truth assumptions, we can save one NFE under very tight budget. At $t=T$, we simply let $\epsilon_\theta(x_T,T)=x_T$, or $x_\theta(x_T,T)=\overline{x_0}$
7. Skip coefficients: This method is inspired by distillation method. We time decay coefficients on the skip connection part in the U-net.

## File structure
The main code of **USF** is contained in `uni_sampler.py`.

We provide examples of using **USF** in `codebases/`. The Latent Diffusion on LSUN Bedroom and Stable Diffusion are implemented in same dir named `codebases/latent_diffusion/`.

## Visual effect

### LSUN Bedroom
<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center;">
    <img src="/images/lsun_nfe3_usf.png" alt="USF" width="300">
    <p>Samples generated by USF NFE=3</p>
  </div>

  <div style="text-align: center;">
    <img src="/images/lsun_nfe3_dpm_solver_v3.png" alt="DPM-Solver-v3" width="300">
    <p>Samples generated by DPM-Solver-v3 NFE=3</p>
  </div>

</div>

### Stable Diffusion
<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center;">
    <img src="images/sd_nfe5_usf.png" alt="USF" width="300">
    <p>Samples generated by USF NFE=5</p>
  </div>

  <div style="text-align: center;">
    <img src="images/sd_nfe5_dpm_solver_v3.png" alt="DPM-Solver-v3" width="300">
    <p>Samples generated by DPM-Solver-v3 NFE=5</p>
  </div>

</div>

## FID results

### ScoreSDE on CIFAR10
| **Method**        | **NFE: 3** | **NFE: 4** | **NFE: 5** | **NFE: 6** | **NFE: 7** | **NFE: 8** | **NFE: 9** | **NFE: 10** |
|--------------------|------------|------------|------------|------------|------------|------------|------------|-------------|
| DPM-Solver++       | 91.24      | 61.37      | 28.39      | 13.41      | 7.44       | 5.33       | 4.37       | 3.99        |
| UniPC              | 93.52      | 60.43      | 23.62      | 10.41      | 6.49       | 5.16       | 4.30       | 3.89        |
| DPM-Solver-v3      | 139.8      | 34.69      | 12.74      | 7.46       | 5.18       | 3.94       | 3.65       | 3.39        |
| **Ours**           | **13.30**  | **6.81**   | **5.40**   | **4.05**   | **3.60**   | **3.00**   | **2.83**   | **2.70**    |
### EDM Model on CIFAR10
| **Method**        | **NFE: 3** | **NFE: 4** | **NFE: 5** | **NFE: 6** | **NFE: 7** | **NFE: 8** | **NFE: 9** | **NFE: 10** |
|--------------------|------------|------------|------------|------------|------------|------------|------------|-------------|
| DPM-Solver++       | 110.71     | 46.47      | 25.14      | 12.31      | 6.96       | 4.61       | 3.51       | 3.08        |
| UniPC              | 110.3      | 44.38      | 24.15      | 11.52      | 6.03       | 4.05       | 3.31       | 2.98        |
| DPM-Solver-v3      | 85.18      | 27.18      | 12.45      | 8.76       | 5.60       | 3.66       | 2.84       | 2.62        |
| **Ours**           | **10.42**  | **6.43**   | **3.93**   | **2.48**   | **2.42**   | **2.43**   | **2.33**   | **2.20**    |
### Guided Diffusion on ImageNet256
| **Method**       | **3**    | **4**    | **5**    | **6**    | **7**    | **8**    | **9**    | **10**   |
|-------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| DPM-Solver++     | 53.14    | 26.3     | 16.97    | 12.92    | 11.04    | 9.88     | 9.14     | 8.75     |
| UniPC            | 53.30    | 25.00    | 15.65    | 11.86    | 10.20    | 9.30     | 8.75     | 8.43     |
| DPM-Solver-v3    | 57.52    | 26.87    | 15.21    | 11.30    | 9.74     | 8.93     | 8.54     | 8.22     |
| Ours             | **32.71**| **18.64**| **13.42**| **10.80**| **9.57** | **8.85** | **8.34** | **8.18** |
### Latent Diffusion on LSUN Bedroom
| **Method**       | **3**   | **4**   | **5**   | **6**   | **7**   | **8**   | **9**   | **10**  |
|-------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| DPM-Solver       | 118.21  | 50.49   | 21.78   | 10.98   | 7.03    | 5.54    | 4.93    | 4.70    |
| DPM-Solver++     | 118.45  | 50.57   | 19.71   | 9.34    | 6.11    | 5.13    | 4.84    | 4.75    |
| UniPC            | 103.23  | 34.62   | 12.36   | 7.51    | 5.81    | 5.20    | 4.86    | 4.73    |
| DPM-Solver-v3    | 81.33   | 29.68   | 12.02   | 6.99    | 5.24    | 4.83    | 4.66    | 4.50    |
| Ours             | **8.62**| **6.65**| **6.04**| **4.81**| **4.49**| **4.12**| **4.13**| **4.14**|
### Stable Diffusion
| **Method**       | **3**    | **4**    | **5**    | **6**    | **7**    | **8**    | **9**    | **10**   |
|-------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| DPM-Solver       | 43.86    | 26.9     | 21.78    | 19.75    | 18.77    | 18.17    | 17.87    | 17.57    |
| DPM-Solver++     | 43.58    | 27.02    | 22.18    | 19.82    | 18.73    | 18.13    | 17.76    | 17.47    |
| UniPC            | 45.62    | 28.3     | 22.44    | 20.05    | 19.04    | 18.52    | 18.00    | 17.74    |
| DPM-Solver-v3    | 59.68    | 31.57    | 22.67    | 19.42    | 18.05    | 17.52    | 17.15    | 16.81    |
| Ours             | **35.06**| **19.51**| **17.66**| **15.32**| **15.38**| **15.22**| **15.47**| **14.95**|