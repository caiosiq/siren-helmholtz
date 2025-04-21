from funcs.common_imports import *
from funcs.siren_structure import *
from funcs.plotting import *
from funcs.define_training import *
"""
Main training script for the SIREN-based PINN model.

This script trains a Physics-Informed Neural Network (PINN) using the SIREN (Sinusoidal Representation Networks) architecture to solve the 2D complex Helmholtz equation with a single point source in free space.

The model is trained on a grid of size `grid_size` with a source position at `source_pos`. The training data is generated using the [generate_training_data](cci:1://file:///c:/Users/CaioV/OneDrive%20-%20Massachusetts%20Institute%20of%20Technology/githubs/siren-helmholtz/funcs/define_training.py:4:0-28:55) function from the `funcs.define_training` module.

The model is trained for `epochs` number of epochs with a learning rate of `lr` and a boundary condition loss weight of `lambda_bc`. The training process is logged every `print_every` epochs.

The trained model is saved to a file named `siren_pinn_trained.pt` in the `models` directory, along with a JSON file containing the model's metadata.

Parameters:
    grid_size (tuple): Size of the grid (N, N)
    source_pos (tuple): Position of the source (x, y)
    physical_size (tuple): Physical size of the domain (Lx, Ly)
    add_dielectric (bool): Whether to add a dielectric material to the domain
    epochs (int): Number of epochs to train the model
    lr (float): Learning rate
    lambda_bc (float): Weight of the boundary condition loss
    print_every (int): Logging frequency

Returns:
    losses (dict): Dictionary containing the training losses
"""


folder = 'reference_paper_larger_more_intensity'
os.makedirs(f"models/{folder}", exist_ok=True)
N = 64
grid_size=(N, N)
source_pos=(20, 20)
physical_size=(N, N)
add_dielectric=True
epochs=100000
lambda_bc=10 #10**3
lr=1e-4
coords, epsilon, source = generate_training_data(grid_size=grid_size, source_pos=source_pos, add_dielectric=add_dielectric,physical_size=physical_size,intensity=100.0)
lambda_src = 10**4



omega = 1.0
hidden_layers = 3
hidden_features = 200
omega_0 = omega*grid_size[0]/physical_size[0]

model = SirenPINN(omega_0=omega_0, hidden_layers=hidden_layers, hidden_features=hidden_features)
# # Set final layer to zero, if source_intensity is zero, the function should have 0 objective function
# with torch.no_grad():
#     model.net[-1].weight.zero_()
#     model.net[-1].bias.zero_()


losses = train(
    model,
    coords,
    epsilon,
    omega,
    source,lambda_src,

    grid_size=grid_size,
    physical_size=physical_size,
    epochs=epochs,
    lr=lr,
    lambda_bc=lambda_bc,
    boundary_type='open',
    print_every=500,
)
save_losses_csv(losses, filename=f"models/{folder}/siren_losses.csv")
torch.save(model.state_dict(), f"models/{folder}/siren_pinn_trained.pt")

metadata = {"grid_size": grid_size,"physical_size": physical_size, "omega": omega, "source_pos": source_pos, "add_dielectric": add_dielectric,"lambda_bc": lambda_bc, "epochs": epochs,"hidden_layers": hidden_layers, "hidden_features": hidden_features, "omega_0": omega_0, "model_path": "siren_pinn_trained.pt", "model_path": "siren_pinn_trained.pt"}
with open(f"models/{folder}/siren_pinn_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
