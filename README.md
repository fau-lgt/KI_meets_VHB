# Topology Optimization meets Machine Learning

This repository contains the code and workflow/exercise to run **graded multiscale topology optimization using neural networks (GM-TOuNN)**, explore designs by varying mesh resolution and target volume fraction, and **reconstruct and validate** optimized designs with **Gmsh** + **PrePoMax/CalculiX**. 

This exercise is part of the "KI meets VHB" funding program of the "Data Acquisition, Processing and Analysis in Manufacturing Engineering and Material Science" lecture at the 'Virtuelle Hochshule Bayern' (vhb).

---

## What this project does

### Goal
Solve topology optimization problems where the objective is to **minimize compliance** (maximize stiffness) under a **material/volume constraint**. 

### What makes GM-TOuNN different
Instead of optimizing element densities directly, GM-TOuNN uses:
- a **library of microstructure unit cells** (e.g., square / xbox / xpbox),
- **homogenized effective properties** for each unit cell,
- and a **neural network** that maps element coordinates `(x, y)` →
  **(unit-cell type, wall thickness / volume fraction control)**.

This reduces cost by avoiding explicit micro-geometry resolution in the global FE mesh and enables fast design-space exploration.

---

## Repository structure

- `main_TOuNN.py` — main file to run a topology optimization
- `config.txt` — problem selection, mesh, training, and saving options
- `VHB_tools.txt` — Python requirements list for the Anaconda environment
- `gmsh_from_predictions.py` — reconstruct a GM-TOUNN predicted design using Gmsh and export as `.inp` file for FEM validation

---

## Prerequisites

### Software
- **Anaconda** (Python distribution & environment manager)
- **PyCharm** (recommended) or VS Code
- **Gmsh** (for reconstruction meshing)
- **PrePoMax** (GUI pre/post for **CalculiX**) to run validation simulations

---
## Citation
This work was developed based on the following work of Chandrasekhar et al.:
```
@article{chand2022GMTOuNN,
author = {Chandrasekhar, Aaditya and Sridhara, Saketh and Suresh, Krishnan},
title = {GM-TOuNN: Graded Multiscale Topology Optimization using  Neural Networks},
journal = {arXiv preprint arXiv:2204.06682},
year={2022}
}
``
