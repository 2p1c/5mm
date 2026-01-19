# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository contains MATLAB scripts for analyzing dispersion curves and RMS imaging of Lamb waves in multi-layer structures. The codebase focuses on processing scanning data (snake scan pattern), theoretical dispersion curve calculation using the Global Matrix Method, and visualization of wave propagation and defects.

## Architecture & Core Components
- **Data Structure**:
  - Raw data is stored in `.mat` files containing time vector `x` and scanning data `y`.
  - Data is reshaped from 2D scanning arrays into 3D matrices `(x_pos, y_pos, time)` for processing.
  - Scanning pattern is "snake-like" (alternating up/down directions for columns).
- **Key Modules**:
  - `dispersion.m` / `dispersion_Theory.m`: Calculates and visualizes frequency-wavenumber dispersion curves for multi-layer plates.
  - `RMS_Imaging.m`: Performs Root Mean Square (RMS) imaging for defect detection.
  - `Filter.m`: Signal processing utilities (bandpass filtering, etc.).
  - `visualization.m`: Utilities for plotting results.

## Key Files
- `dispersion.m`: Main script for experimental dispersion analysis.
- `dispersion_Theory.m`: Theoretical calculation of dispersion curves using Global Matrix Method.
- `RMS_Imaging.m`: Generates RMS images from scan data to visualize defects.
- `processed_data.mat`: Intermediate storage for processed signal data.

## Development Workflow
Since this is a MATLAB project, standard CLI build/test commands are not applicable.
- **Running Scripts**: Scripts are intended to be run within the MATLAB environment.
- **Data Paths**: Scripts use absolute paths (e.g., `C:\Users\123\Documents\Projects\5mm\data\...`) or relative paths to `data/`. When refactoring, prefer relative paths or configurable bases.
- **Dependencies**: Most scripts depend on standard MATLAB toolboxes (Signal Processing, etc.).

## Code Style & Conventions
- **Comments**: Code is commented in Chinese. Maintain this convention for consistency.
- **Matrix Operations**: Vectorized operations are preferred over loops where possible for performance.
- **Visualization**: Figures typically use `imagesc` for field data and standard plots for curves.

## Copilot/AI Instructions
- **Language**: Use Chinese for comments and explanations within code files.
- **Path Handling**: Be aware of hardcoded paths in existing scripts; suggest robust path handling (e.g., `fullfile`) when modifying.
- **Data Loading**: Ensure data reshaping logic correctly handles the snake scan pattern (flipping even/odd columns).
