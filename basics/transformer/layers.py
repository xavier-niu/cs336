import math

from einops import einsum
import torch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # init weights on device by Xavier/Glorot init method
        # shape is (d_out, d_in)
        weight_tensor = torch.empty((out_features, in_features), dtype=dtype, device=device)
        self.weight = torch.nn.Parameter(weight_tensor)
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        - x shape is (..., d_in), where d_in, in terms of LM's first FFN
        d_model == size of embeddings
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
