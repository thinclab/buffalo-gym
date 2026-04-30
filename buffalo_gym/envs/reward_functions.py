import torch


class Polynomial(torch.nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.coefficients = torch.tensor(coefficients, dtype=torch.float32)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.sum(self.coefficients * (x ** torch.arange(len(self.coefficients))), dim=1, keepdim=True)


class Gaussian(torch.nn.Module):
    def __init__(self, mus, alpha=40.0):
        super().__init__()
        self.register_buffer("mus", torch.tensor(mus))
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)

        return torch.sum(torch.exp(-self.alpha * (x - self.mus) ** 2), dim=1, keepdim=True)
