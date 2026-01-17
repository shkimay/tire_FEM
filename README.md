# The 5th KAIST-POSTECH-UNIST AI & Data Science Competition

## Overview
**Subject**: Development of a Tire Defect Prediction Model Based on FEM Simulation Data

**Period**: Nov 2025 – Dec 2025 (Preliminary Round)


**(Task 1)** Predicting tire defect rates using Finite Element Method (FEM)–based simulation results
**(Task 2)** Making optimal pilot production decisions based on those predictions

## Data
- FEM simulation data: $$(x,y,p)$$ at 256 points
- Tabular data: process parameters and design-related attributes

## Feature Engineering
**1. Grid-based FEM Depth Representation**
   - FEM pressure values were interpolated onto a 2D grid to construct depth maps based on pressure $p$.
**2. Laplacian-KNN Statistics**
   - Local pressure variations were captured by computing KNN-based Laplacian statistics over FEM points. 

## Models
Tree-based models: CatBoost, XGBoost
Deep model: Swin Transformer
Ensemble strategy: Out-of-fold (OOF) probability-based ensemble with AUC-oriented weighting
