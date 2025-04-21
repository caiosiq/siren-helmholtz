from funcs.common_imports import *
from funcs.loss_func import *
from funcs.siren_structure import *
# 6. Generate the training data
def generate_training_data(grid_size=(256, 256), source_pos=(128, 128), add_dielectric=False, physical_size=(100.0, 100.0), device="cpu",intensity=1.0):
    Nx, Ny = grid_size
    Lx, Ly = physical_size

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)

    epsilon = np.ones((Nx, Ny))
    if add_dielectric:
        cx, cy = Lx / 2, Ly / 2
        r = Lx / 10
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        epsilon[mask] = 2.0

    source = np.zeros((Nx, Ny), dtype=np.complex64)
    i, j = source_pos
    source[i, j] = intensity

    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    epsilon_tensor = torch.tensor(epsilon.ravel(), dtype=torch.float32, device=device)
    source_tensor = torch.tensor(source.ravel(), dtype=torch.complex64, device=device)

    return coords_tensor, epsilon_tensor, source_tensor
# 7. Training loop
def train(model, coords, epsilon, omega, source,lambda_src, grid_size,physical_size, epochs=5000, lr=1e-4, lambda_bc=1.0, batch_size=8192, print_every=500,boundary_type='open'):
    print("Nonzero source:", torch.max(torch.abs(source)).item())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    coords = coords.to(device)
    epsilon = epsilon.to(device)
    source = source.to(device)
    boundary_coords = extract_boundary_coords(grid_size, physical_size, device=device)

    N = coords.shape[0]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5000, min_lr=lr/100, verbose=True
    )

    total_losses = []
    pde_losses = []
    bc_losses = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # 🔁 Random sample indices
        # idx = torch.randperm(N)[:batch_size]
        # coords_batch = coords[idx]
        # epsilon_batch = epsilon[idx]
        # source_batch = source[idx]
        if batch_size==-1 or batch_size>=N:
            coords_batch = coords
            epsilon_batch = epsilon
            source_batch = source

        # Compute loss on the sampled batch
        total_loss, loss_pde, loss_bc = compute_loss(
            model,boundary_type, coords_batch, epsilon_batch, omega, source_batch,lambda_src,
            lambda_bc=lambda_bc, boundary_coords=boundary_coords
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        # Log
        total_losses.append(total_loss.item())
        pde_losses.append(loss_pde.item())
        bc_losses.append(loss_bc.item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:>5}] Total Loss: {total_loss.item():.3e} | PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e}")

    return {
        "total": total_losses,
        "pde": pde_losses,
        "bc": bc_losses
    }

def save_losses_csv(losses_dict, filename="losses.csv"):
    import csv
    keys = list(losses_dict.keys())
    rows = zip(*[losses_dict[k] for k in keys])  # transpose

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)  # header
        writer.writerows(rows)
def load_losses_csv(filename="siren_losses.csv"):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        headers = rows[0]
        data = rows[1:]

        # Convert strings to floats
        losses_dict = {key: [] for key in headers}
        for row in data:
            for i, val in enumerate(row):
                losses_dict[headers[i]].append(float(val))
                
        return losses_dict