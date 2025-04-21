from funcs.common_imports import *
# 4. Define the Helmholtz PDE residual
def pde_residual(model, coords, epsilon, omega, source, lambda_src, normalize=False, form='L1'):
    """
    Computes the PDE residual loss with optional normalization and source emphasis.
    """
    assert form in ['L1', 'L2'], f"Invalid loss form: {form}"
    coords.requires_grad_(True)
    device = coords.device

    output = model(coords)
    Re_Ez = output[:, 0]
    Im_Ez = output[:, 1]
    Ez = Re_Ez + 1j * Im_Ez

    grads = torch.autograd.grad(Ez, coords, grad_outputs=torch.ones_like(Ez), create_graph=True)[0]
    dEz_dx = grads[:, 0]
    dEz_dy = grads[:, 1]

    d2Ez_dx2 = torch.autograd.grad(dEz_dx, coords, grad_outputs=torch.ones_like(dEz_dx), create_graph=True)[0][:, 0]
    d2Ez_dy2 = torch.autograd.grad(dEz_dy, coords, grad_outputs=torch.ones_like(dEz_dy), create_graph=True)[0][:, 1]

    laplacian_Ez = d2Ez_dx2 + d2Ez_dy2

    # Helmholtz residual
    residual = (-laplacian_Ez - epsilon * (omega**2) * Ez) + 1j * omega * source

    if normalize:
        Ez_mag = torch.abs(Ez+1j*omega*source+1e-8)
        scaling = Ez_mag + 1e-8  # avoid div by 0
    else:
        scaling = 1.0
    if form == 'L1':
        loss = torch.abs(residual) / scaling
    elif form == 'L2':
        loss = (torch.abs(residual) ** 2) / (scaling ** 2)

    # Build masks
    mask_src = (torch.abs(source) > 1e-8).float()
    mask_rest = 1.0 - mask_src

    # Weighted loss
    loss_src = (loss * mask_src).sum() 
    loss_rest = (loss * mask_rest).sum() 
    return (lambda_src * loss_src + loss_rest)

# def boundary_loss(model, coords, grid_size,omega):
#     """
#     Enforces Dirichlet boundary condition E_z = 0 on all domain boundaries.
    
#     Parameters:
#         model      : the PINN model
#         coords     : tensor of all (x, y) points, shape (N, 2)
#         grid_size  : tuple (Nx, Ny) describing the 2D grid shape

#     Returns:
#         loss       : scalar, mean squared error at the boundary
#     """
#     Nx, Ny = grid_size
#     coords_2d = coords.view(Nx, Ny, 2)  # reshape to grid

#     # Collect boundary points (top, bottom, left, right)
#     top    = coords_2d[0, :, :]        # y = 0
#     bottom = coords_2d[-1, :, :]       # y = 1
#     left   = coords_2d[:, 0, :]        # x = 0
#     right  = coords_2d[:, -1, :]       # x = 1

#     boundary_coords = torch.cat([top, bottom, left, right], dim=0)

#     # Model prediction
#     output = model(boundary_coords)  # shape (B, 2)
#     Re_Ez = output[:, 0]
#     Im_Ez = output[:, 1]

#     # Penalize non-zero values
#     loss = torch.mean(Re_Ez**2 + Im_Ez**2)
#     return loss

def extract_boundary_coords(grid_size, physical_size, device="cpu"):
    Nx, Ny = grid_size
    Lx, Ly = physical_size
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    coords = []
    normals = []

    for i in range(Nx):
        coords.append([x[i], 0.0])
        normals.append([0.0, -1.0])
        coords.append([x[i], Ly])
        normals.append([0.0, 1.0])
    for j in range(Ny):
        coords.append([0.0, y[j]])
        normals.append([-1.0, 0.0])
        coords.append([Lx, y[j]])
        normals.append([1.0, 0.0])

    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    normals_tensor = torch.tensor(normals, dtype=torch.float32, device=device)
    return torch.cat([coords_tensor, normals_tensor], dim=1)  # shape (N, 4)


def boundary_loss_open(model, boundary_coords, omega):
    """
    Applies Sommerfeld condition ∂E/∂n + iωE = 0 at provided boundary points.
    Assumes normals are encoded along with each boundary point.
    """
    device = boundary_coords.device
    coords = boundary_coords[:, :2]
    normals = boundary_coords[:, 2:]

    coords.requires_grad_(True)
    output = model(coords)
    Re_Ez = output[:, 0]
    Im_Ez = output[:, 1]
    Ez = Re_Ez + 1j * Im_Ez

    grads = torch.autograd.grad(Ez, coords, grad_outputs=torch.ones_like(Ez), create_graph=True)[0]
    dE_dn = (grads * normals).sum(dim=1)  # directional derivative along normal

    residual = dE_dn + 1j * omega * Ez
    return (torch.abs(residual)**2).sum()/torch.mean(torch.abs(Ez)**2)

def boundary_loss_none(model, boundary_coords, omega):
    """
    Applies Sommerfeld condition ∂E/∂n + iωE = 0 at provided boundary points.
    Assumes normals are encoded along with each boundary point.
    """
    device = boundary_coords.device
    coords = boundary_coords[:, :2]
    normals = boundary_coords[:, 2:]

    coords.requires_grad_(True)
    output = model(coords)
    Re_Ez = output[:, 0]
    Im_Ez = output[:, 1]
    Ez = Re_Ez + 1j * Im_Ez

    grads = torch.autograd.grad(Ez, coords, grad_outputs=torch.ones_like(Ez), create_graph=True)[0]
    dE_dn = (grads * normals).sum(dim=1)  # directional derivative along normal

    residual = dE_dn + 1j * omega * Ez
    return torch.mean(torch.abs(residual)**2)


# 5. Define the loss function (PDE + boundary + source loss)
def compute_loss(model,boundary_type, coords, epsilon, omega, source,lambda_src, lambda_bc=1.0, boundary_coords=None):
    loss_pde = pde_residual(model, coords, epsilon, omega, source,lambda_src)
    loss_bc = 0.0
    if boundary_coords is not None:
        if boundary_type == 'open':
            loss_bc = boundary_loss_open(model, boundary_coords, omega)
        elif boundary_type == 'none':
            loss_bc = boundary_loss_none(model, boundary_coords, omega)
    total_loss = loss_pde + lambda_bc * loss_bc
    return total_loss, loss_pde, loss_bc


    




