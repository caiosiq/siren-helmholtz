#9. Test the model
from funcs.common_imports import *
from funcs.loss_func import *
from funcs.siren_structure import *
from funcs.define_training import *
def test_siren_model():
    model = SirenPINN()
    xy = torch.rand((10, 2))  # 10 random (x, y) points
    out = model(xy)
    assert out.shape == (10, 2), "Output shape should be (N, 2) for real and imag Ez"
    print("✅ SIREN model forward pass test passed.")

def test_grid_generation():
    coords, epsilon, source = generate_training_data(grid_size=(64, 64), source_pos=(32, 32), add_dielectric=True)
    assert coords.shape == (64 * 64, 2)
    assert epsilon.shape == (64 * 64,)
    assert source.shape == (64 * 64,)
    print("✅ Grid generation test passed.")

def test_pde_residual():
    coords, epsilon, source = generate_training_data(grid_size=(32, 32), source_pos=(16, 16))
    model = SirenPINN()
    omega = 1.0
    loss = pde_residual(model, coords, epsilon, omega, source)
    assert torch.isfinite(loss).item(), "PDE residual returned NaN or Inf"
    print("✅ PDE residual test passed.")

def test_boundary_loss():
    coords, epsilon, source = generate_training_data(grid_size=(32, 32), source_pos=(16, 16))
    model = SirenPINN()
    loss = boundary_loss_open(model, coords, grid_size=(32, 32),omega=1.0)
    assert torch.isfinite(loss).item(), "Boundary loss returned NaN or Inf"
    print("✅ Boundary condition test passed.")

def test_full_loss():
    coords, epsilon, source = generate_training_data(grid_size=(32, 32), source_pos=(16, 16))
    model = SirenPINN()
    omega = 1.0
    total_loss, loss_pde, loss_bc = compute_loss(model, coords, epsilon, omega, source, grid_size=(32, 32))
    assert torch.isfinite(total_loss).item(), "Total loss returned NaN or Inf"
    print(f"✅ Full loss test passed: PDE={loss_pde.item():.3e}, BC={loss_bc.item():.3e}")


