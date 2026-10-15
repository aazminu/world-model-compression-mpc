# World-Model Compression & MPC Operator Distillation (`world-model-compression-mpc`)

Systematic framework for compressing latent dynamics world models used inside closed-loop **Cross-Entropy Method (CEM)** Model Predictive Control (MPC) planners.

## Core Contributions

1. **Weight SVD vs. Activation-Aware SVD (`asvd_compression.py`)**:
   - Implements low-rank factorization of latent predictor matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with parameter-budget rank selection $r = \lfloor \frac{\rho \cdot d_{\text{out}} d_{\text{in}}}{d_{\text{out}} + d_{\text{in}}} \rfloor$.
   - Scales columns by empirical latent activation second moments $S_{jj} = (\mathbb{E}[x_j^2])^{\alpha/2}$ prior to truncated SVD, reducing multi-step rollout compounding error on high-energy latent dimensions.
2. **Closed-Loop CEM Planner on `TwoRoom` and `PushT` (`cem_mpc_planner.py`)**:
   - Evaluates how open-loop MSE vs. elite-set trajectory ordering affects closed-loop task success across bottleneck doorway navigation (`TwoRoom`) and planar contact manipulation (`PushT`).
3. **Cache-Based Operator Distillation (`operator_distillation.py`)**:
   - Compares **Forward KL** distillation against **Hybrid Forward KL + Elite-Classification**, aligning the student world model's top-$K$ elite trajectory rankings with the uncompressed teacher planner.
