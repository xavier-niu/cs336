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
        self.weights = torch.nn.Parameter(weight_tensor)
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        - x shape is (..., d_in), where d_in, in terms of LM's first FFN
        d_model == size of embeddings
        """
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors, i.e.
                d_model.
            device (torch.device | None): Device to store the parameters
                on. Defaults to None (PyTorch's default device).
            dtype (torch.dtype | None): Data type of the parameters.
                Defaults to None (PyTorch's default dtype).
        """
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs

        Args:
            token_ids (torch.Tensor): Token IDs vector in shape of (...)

        Returns:
            torch.Tensor: Embeddings in shape of (..., d_model)
        """
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # init gain parameters to ones
        # shape: d_model
        self.gains = torch.nn.Parameter(torch.empty((d_model,), dtype=dtype, device=device))
        torch.nn.init.ones_(self.gains)

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the tensor x

        Args:
            x (torch.Tensor): A tensor of shape
                `(batch_size, seq_len, d_model)`

        Returns:
            torch.Tensor: A tensor that has a same shape of x
        """
        # upcast to float32 to prevent overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # (batch_size, seq_len, d_model)
        x_sq = torch.square(x)
        # (batch_size, seq_len, 1)
        mean_sq = torch.mean(x_sq, dim=-1, keepdim=True)
        # (batch_size, seq_len, 1)
        rms = torch.sqrt((mean_sq+self.eps))
        # (batch_size, seq_len, d_model)
        ret = x / rms * self.gains

        return ret.to(in_dtype)
