from funcs.common_imports import *
from funcs.siren_structure import *
def plot_siren_output_before_training(model, grid_size, physical_size=(100.0, 100.0), title_prefix="", save_dir=None):
    """
    Plot the real and imaginary parts of the SIREN output before training, including FFT spectrum of Re(Ez).
    """
    Nx, Ny = grid_size
    Lx, Ly = physical_size

    # Generate coordinates
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        output = model(coords_tensor)
        Re_Ez = output[:, 0].numpy().reshape(Nx, Ny)
        Im_Ez = output[:, 1].numpy().reshape(Nx, Ny)

    # Compute FFT of Real(Ez)
    Ez_fft = np.fft.fftshift(np.abs(np.fft.fft2(Re_Ez)))
    Ez_fft_log = np.log1p(Ez_fft)

    # Plot all 3 views
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    im = ax.imshow(Re_Ez, cmap="RdBu", origin="lower", extent=[0, Lx, 0, Ly])
    ax.set_title(f"{title_prefix}Untrained Real(Ez)")
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(Im_Ez, cmap="RdBu", origin="lower", extent=[0, Lx, 0, Ly])
    ax.set_title(f"{title_prefix}Untrained Imag(Ez)")
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(Ez_fft_log, cmap="inferno", origin="lower")
    ax.set_title("log(FFT) of Real(Ez)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_untrained_siren_output.png"))

    plt.show()

omega_0=1
model = SirenPINN(omega_0=omega_0, hidden_layers=4, hidden_features=64)
plot_siren_output_before_training(model, grid_size=(256, 256), physical_size=(128.0, 128.0), title_prefix=f"ω0={omega_0}", save_dir="results/sanity")