# Symbolic Regression Genetic Model ReadMe (rought draft)

## Overview

The **Symbolic Regression Genetic Modell** is a machine learning project that integrates symbolic regression with deep learning. It involves generating symbolic equations for a dataset, optimizing these equations using simplification techniques, and comparing their performance with neural network latent representations.

### Key Features:
- **Symbolic Regression**: The project uses symbolic equations generated from data.
- **Optimization**: Equations are optimized using recursive decomposition and SymPy-based simplification techniques.
- **Comparison**: The original and optimized equations are compared in terms of complexity, execution time, and accuracy.
- **Autoencoder**: A deep learning autoencoder is used to compute latent dimensions for performance evaluation.
- **Safe Operations**: The project uses custom safe operators (e.g., `safe_log`, `safe_sqrt`) to handle numerical stability during equation evaluation.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/lawful_induction_model.git
   cd lawful_induction_model

2. **Create a virtual environment (optional but recommended)** 
   python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

3.	**Install the required packages:**
pip install -r requirements.txt

4.	Install SymPy and PyTorch dependencies (if not included in requirements.txt):
    pip install sympy torch

5. Run the script: python main.py to run the full training and genetic loop. Run python run_pretrianed to execute a pretrained model and collect metrics off it 

## Project Structure
• main.py: runs the main training loop
•	run_pretrained.py: runs the pretrained model indepdently of the training loop 
•	data.py: Contains the data generator for the California Housing dataset.
•	autoencoder.py: Defines the autoencoder architecture.
•	optimize_equations.py: (wip) Contains the symbolic equation optimization and comparison functions.
•	requirements.txt: Lists all required dependencies.