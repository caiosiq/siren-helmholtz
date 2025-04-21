📡 SIREN-Based PINN for Solving the 2D Complex Helmholtz Equation
This project implements a Physics-Informed Neural Network (PINN) using the SIREN (Sinusoidal Representation Networks) architecture to solve the 2D Helmholtz equation with a single point source in free space.

🔬 Overview
This simulation was developed as part of a technical challenge and explores how SIREN architectures can model wave-like solutions governed by the complex Helmholtz equation. The solution is expressed as a complex field Ez = Re(Ez) + i·Im(Ez), and the neural network is trained to minimize the PDE residual, boundary conditions, and source error.

Key features:

✅ Custom SIREN layers with ω₀ scaling

✅ Separate prediction and training of Re(Ez) and Im(Ez)

✅ Sommerfeld boundary condition: ∂E/∂n + i·ωE = 0

✅ Real-valued point source (Jz) represented as a single delta-like numerical spike

✅ Highly modular and easy to expand with additional geometries or media properties



🧠 Model Details
Architecture: SIREN PINN with sine activations

ω₀: Matches the physical frequency scale of the field

Domain: 2D square domain [0, 64] x [0, 64]

Loss Function:

PDE loss on complex Helmholtz residual

Weighted boundary condition loss (Sommerfeld)

Scaled error at the source location (to amplify learning)

Training Strategy:

Epochs: 100,000+

LR decay every 5000 epochs

L1 loss preferred over L2 for convergence



🗂️ Project Structure
siren-helmholtz/

│

├── funcs/                  # Common utility functions (loss, physics, geometry)

├── models/                # Definition of the Siren layers and model

├── own_model/             # Core script for building and training SIREN PINNs

├── results/               # Saved predictions and output plots

├── train_model.py         # Main training script

├── read_model.py          # Script to load and visualize saved models

├── test.py                # Helper tests

├── hyper_parameters_analysis.py  # Sweep script to study convergence

📈 Sample Results
Below is an example showing Re(Ez) and Im(Ez) after 100,000 training steps. The source is located at (20, 20). The model successfully minimizes Re(Ez) and localizes the field to the imaginary component, as expected for a real Jz source.



🚀 Setup & Usage
You can clone and run the code with:
git clone https://github.com/YOUR_USERNAME/siren-helmholtz.git
cd siren-helmholtz
pip install -r requirements.txt
python train_model.py

To visualize the saved results:
python read_model.py


🤔 Future Work

GPU-accelerated scaling for higher grid sizes
Running simulation till higher Epochs
Integration with DeepXDE-style autograd backend for faster experiments


📬 Acknowledgements
This work was originally developed as part of a challenge by KronosAI. Thanks to Ziyi Yin, PhD, for the opportunity and feedback.

💡 Questions?
Reach out to me at caiosiq@mit.edu or open an issue on this repo!


🧠 References

SIREN Paper (Sitzmann et al.)
DeepXDE Framework
