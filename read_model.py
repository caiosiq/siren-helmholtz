from funcs.common_imports import *
from funcs.siren_structure import *
from funcs.plotting import *
from funcs.define_training import *

"""
Script to load and visualize saved SIREN PINN models.

This script loads a trained SIREN PINN model from a file, along with its metadata, and visualizes the predicted fields (Re(Ez) and Im(Ez)).

Parameters:
    folder (str): Name of the folder containing the saved model and metadata

Returns:
    None
"""

folder = 'reference_paper_larger_more_intensity'
with open(f"models/{folder}/siren_pinn_metadata.json", "r") as f:
    metadata = json.load(f)

print("Loaded metadata:")
print(metadata)
#Load dict CSV file of losses
losses = load_losses_csv(filename=f"models/{folder}/siren_losses.csv")

grid_size = metadata["grid_size"]
source_pos = metadata["source_pos"]
add_dielectric = metadata["add_dielectric"]
coords, epsilon, source = generate_training_data(grid_size=grid_size, source_pos=source_pos, add_dielectric=add_dielectric)


omega = metadata["omega"]
hidden_layers = metadata["hidden_layers"]
hidden_features = metadata["hidden_features"]
omega_0 = omega/hidden_layers
model = SirenPINN(omega_0=omega_0, hidden_layers=hidden_layers, hidden_features=hidden_features)

# model.load_state_dict(torch.load(f"models/{folder}/siren_pinn_trained.pt"))
model.load_state_dict(torch.load(f"models/{folder}/siren_pinn_trained.pt", weights_only=True))

model.eval()
plot_input_map(grid_size=grid_size, epsilon=epsilon, source=source,show=False, save_dir=f"results/{folder}")
plot_results(model, coords, grid_size=grid_size, title_prefix="ω=1.0 ", show=False, save_dir=f"results/{folder}")
plot_loss_curve(losses,show=False, save_dir=f"results/{folder}")
# print(losses)