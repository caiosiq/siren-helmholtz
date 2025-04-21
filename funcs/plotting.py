from funcs.common_imports import *
# 8. Visualization
def plot_input_map(grid_size, epsilon, source, show=True, save_dir="."):
    Nx, Ny = grid_size
    epsilon_2d = epsilon.view(Nx, Ny).numpy()
    source_2d = source.abs().view(Nx, Ny).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(epsilon_2d, cmap='viridis')
    plt.title("Permittivity Map ε(x, y)")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(source_2d, cmap='hot')
    plt.title("Source |Jz(x, y)|")
    plt.colorbar()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'inputmap.png'))

    if show:
        plt.show()
    else:
        plt.close()

def plot_results(model, coords, grid_size, title_prefix="", show=True, save_dir="."):
    model.eval()
    with torch.no_grad():
        output = model(coords)
        Re_Ez = output[:, 0].cpu().numpy()
        Im_Ez = output[:, 1].cpu().numpy()

    Nx, Ny = grid_size
    Re_Ez_2D = Re_Ez.reshape(Nx, Ny)
    Im_Ez_2D = Im_Ez.reshape(Nx, Ny)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(Re_Ez_2D, cmap="RdBu", origin="lower")
    ax.set_title(f"{title_prefix}Real(Ez)")
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(Im_Ez_2D, cmap="RdBu", origin="lower")
    ax.set_title(f"{title_prefix}Imag(Ez)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'results.png'))

    if show:
        plt.show()
    else:
        plt.close()
    
def plot_loss_curve(losses_dict, show=True, save_dir="."):
    epochs = range(1, len(losses_dict["total"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses_dict["total"], label="Total Loss")
    plt.plot(epochs, losses_dict["pde"], label="PDE Loss")
    plt.plot(epochs, losses_dict["bc"], label="Boundary Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Losses Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))

    if show:
        plt.show()
    else:
        plt.close()