# PINN-1D-Viscoelastic-Wave-Equation

Physics‑Informed Neural Network (PINN) implementation solving the 1D viscoelastic wave equation using [DeepXDE](https://github.com/lululxvi/deepxde) and PyTorch.  
Includes both **forward problem** solving and **inverse parameter identification**.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![DeepXDE](https://img.shields.io/badge/DeepXDE-1.0+-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Governing Equation](#governing-equation)
  - [Analytical Solution](#analytical-solution)
  - [PINN Approach](#pinn-approach)
  - [Forward Analysis](#forward-analysis)
  - [Inverse Problem](#inverse-problem)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
---

## Overview

This repository implements a Physics‑Informed Neural Network (PINN) to solve the 1D viscoelastic wave equation. The project aims to simulate wave propagation under an **underdamped regime**, using the analytical solution as a ground truth to validate the predictive accuracy of the neural network model.

---

## Methodology

### Governing Equation

The physical problem is governed by the 1D viscoelastic wave equation:

$$
\rho \ u_{tt} = E \ u_{xx} + \eta \ u_{xxt}
$$

where:

- $\rho$ – density  
- $E$ – elastic modulus  
- $\eta$ – viscosity  

*(Equation 5.94 from Kramer’s *Geotechnical Earthquake Engineering* book)*

### Analytical Solution

The underdamped analytical solution *(Section 5.5.1 – Material Damping Kramer's *Geotechnical Earthquake Engineering* book)* is used as the ground truth for validation.

### PINN Approach

The model uses the **DeepXDE** library with a **PyTorch** backend.

- **Architecture**: Feed‑forward Neural Network (FNN)  
- **Hidden layers**: 4 layers, 50 neurons each  
- **Activation**: Hyperbolic tangent (`tanh`)  
- **Initialization**: Glorot uniform  
- **Optimisation**: Two‑stage (Adam → L‑BFGS) with soft adaptive loss weighting and PDE point resampling.

### Forward Analysis

The forward problem solves for **known** parameters:

- $E = 4.0$
- $\eta = 0.1$
- $\rho = 1.0$

### Inverse Problem

The inverse problem recovers the **unknown** material parameters $E$ and $\eta$ as trainable variables, optimised jointly with the solution field.

---

## Repository Structure
- **`Code/`** – contains the core notebook implementing geometry, BCs/ICs, and training loops.  
- **`Results/`** – output directory for visualisations (solution maps, point‑wise error plots).  
- **`Checkpoint/`** – stores training variables and history.

---

## Results

### 1. Forward Problem

Trained using the PDE and initial/boundary conditions, validated against the analytical solution.

| Metric              | Value    |
|---------------------|----------|
| **Train Loss**      | 1.16e‑05 |
| **Test Loss**       | 1.12e‑05 |
| **Relative L2 Error** | 3.19e‑01 |

### 2. Inverse Problem

Recovered parameters $E$ and $\eta$ with remarkable accuracy:

| Parameter | True Value | Recovered | Relative Error |
|-----------|------------|-----------|----------------|
| $E$       | 4.00       | ~4.00     | 0.02 %         |
| $\eta$    | 0.10       | ~0.10     | 0.01 %         |

---

## Requirements

- `deepxde`
- `torch`
- `numpy`
- `matplotlib`

---
## Installation

Install all dependencies via pip:

```bash
pip install deepxde torch numpy matplotlib jupyter
```

## Usage

### 1. Configure the Backend
Set the DeepXDE backend to PyTorch before importing DeepXDE:

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
```

### 2. Launch the Notebook
Navigate to your code directory and open the example template:

```bash
cd Code
jupyter notebook PINN_Example_13.ipynb
```

### 3. Configure Output Path
Update the output directory inside the notebook to your preferred path:

```python
OUTDIR = "/your/local/path/"
```
