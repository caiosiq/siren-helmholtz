from funcs.common_imports import *
from funcs.siren_structure import *
from funcs.plotting import *
from funcs.define_training import *
omega = 1.0
N = 128
grid_size=(N, N)
source_pos=(N//2, N//2)
physical_size=(1.0*N, 1.0*N)
add_dielectric=False
coords, epsilon, source = generate_training_data(grid_size=grid_size, source_pos=source_pos, add_dielectric=add_dielectric,physical_size=physical_size)
class DummyAnalyticModel(torch.nn.Module):
    def forward(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        Ez_real = torch.sin(omega * x) * torch.sin(omega * y)
        Ez_real = torch.sin(omega * x)
        Ez_imag = torch.zeros_like(Ez_real)
        return torch.stack([Ez_real, Ez_imag], dim=1)

model = DummyAnalyticModel()
loss=  pde_residual(model, coords, epsilon, omega, source, lambda_src=0.1, normalize=True)
print(loss)


def debug_pde_residual(model, coords, epsilon, omega):
    coords.requires_grad_(True)

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
    residual = -laplacian_Ez - epsilon * (omega**2) * Ez  # No source

    res_sq = torch.abs(residual)**2
    return torch.mean(res_sq)

model = DummyAnalyticModel()
loss = debug_pde_residual(model, coords, epsilon, omega)
print("DEBUG loss:", loss.item())