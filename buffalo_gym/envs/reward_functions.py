import torch


class Polynomial(torch.nn.Module):
    def __init__(self, coefficients, powers):
        super().__init__()
        self.register_buffer("coefficients", torch.tensor(coefficients))
        self.register_buffer("powers", torch.tensor(powers))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.sum(self.coefficients * (x**self.powers), dim=1, keepdim=True)


class Gaussian(torch.nn.Module):
    def __init__(self, mus, alphas, coefs, norm):
        super().__init__()
        self.register_buffer("mus", torch.tensor(mus))
        self.register_buffer("alphas", torch.tensor(alphas))
        self.register_buffer("coefs", torch.tensor(coefs))
        self.register_buffer("norm", torch.tensor(norm))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)

        return torch.sum(self.coefs * torch.exp(-self.alphas * (x - self.mus) ** 2), dim=1, keepdim=True) / self.norm
