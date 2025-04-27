# Structured Products in Python

A comprehensive Python library for pricing, analyzing, and managing structured financial products and derivatives.

## Overview

This project provides a robust framework for working with structured financial products, including options pricing, bonds. The main interface is provided through `app.py`.

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main application interface
- `src/`: Core library modules
  - `bonds.py`: Bond pricing and analytics
  - `options/`: Option pricing models (Black-Scholes, Monte Carlo, Trees)
  - `swaps.py`: Swap contract implementations
  - `time_utils/`: Date and time utilities
- `doc/`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Produit_Structure.git

# Navigate to project directory
cd Produit_Structure

# Install dependencies
pip install -r requirements.txt
```