import torch


# Math utils
EPS = 1e-7
def saturate(x, low=0.0, high=1.0):
    return x.clip(low, high)

def safe_exp(x):
    """The same as torch.exp(x), but clamps the input to prevent NaNs."""
    # return torch.exp(torch.minimum(x, torch.ones_like(x) * 87.5))
    return torch.exp(x)

def magnitude(x: torch.Tensor) -> torch.Tensor:
    return safe_sqrt(dot(x, x))

def dot(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1, keepdims=True)

def safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(torch.maximum(x, torch.ones_like(x) * EPS))

def safe_div(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.divide(x1, torch.maximum(x2, torch.ones_like(x2) * EPS))

def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    sqrt_in = torch.maximum(x, torch.ones_like(x) * EPS)
    return torch.sqrt(sqrt_in)

def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return d - 2 * dot(d, n) * n

def mix(x, y, a):
    a = a.clip(0, 1)
    return x * (1 - a) + y * a