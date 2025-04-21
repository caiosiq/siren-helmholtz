from funcs.common_imports import *

# 2. Define the SIREN Layer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: initialize uniformly in [-1/in, 1/in]
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers: initialize with scaling for omega_0
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                     math.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# 3. Define the Full SIREN Network
class SirenPINN(nn.Module):
    def __init__(self, in_features=2, hidden_features=64, hidden_layers=4, out_features=2, omega_0=30):
        super().__init__()
        layers = []

        # First layer: special initialization
        layers.append(SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))

        # Final layer: linear to output (no sine)
        final_layer = nn.Linear(hidden_features, out_features)

        # Initialize final layer
        with torch.no_grad():
            final_layer.weight.uniform_(-math.sqrt(6 / hidden_features) / omega_0,
                                         math.sqrt(6 / hidden_features) / omega_0)
        layers.append(final_layer)

        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
