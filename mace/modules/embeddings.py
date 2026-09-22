from typing import Any, Dict, Optional

import torch
from torch import nn


def _zero_last_layer(embedder: nn.Module) -> None:
    """Zero the parameters of ``embedder``'s output layer, in place.

    Both weight and bias go to zero: a zeroed weight alone still emits the bias,
    which would break the identity the caller asked for.
    """
    if isinstance(embedder, nn.Embedding):
        last: nn.Module = embedder
    elif isinstance(embedder, nn.Sequential):
        last = embedder[-1]
    else:  # pragma: no cover - only the two constructions above are built here
        raise TypeError(f"cannot zero-initialise embedder of type {type(embedder)}")
    nn.init.zeros_(last.weight)
    if getattr(last, "bias", None) is not None:
        nn.init.zeros_(last.bias)


class GenericJointEmbedding(nn.Module):
    """
    Simple concat‐fusion of any set of node‐ or graph‐level features
    with a base embedding.  All features are embedded (via Embedding or small MLP),
    then concatenated onto `species_emb`, passed through SiLU+Linear.

    Two optional per-spec keys support conditioning on a physical scalar:

    ``scale``
        A factor applied to a continuous feature before it is embedded, so the
        model can be fed the quantity in its natural units. An electronic
        temperature in Kelvin uses ``scale: 1.0e-4``, putting the training range
        near unity without anything outside the model having to remember the
        convention: the factor is part of ``embedding_specs`` and therefore
        travels in the checkpoint, so a calculator and a training run cannot
        silently disagree about it.

    ``zero_init``
        Zero the final layer of that feature's embedder, so the feature
        contributes exactly nothing until it is trained. With a single spec the
        concatenation is then zero, the bias-free ``project`` maps it to zero,
        and ``SiLU(0) = 0``, so a freshly built conditioned model reproduces its
        unconditioned counterpart bit for bit at every value of the feature.

        Gradient reaches the zeroed layer immediately (it is multiplied by
        ``project``'s weights, which are not zero) but not ``project`` itself on
        the first step, since that gradient is proportional to the zeroed
        activations. This is inherent to zero-initialising a multiplicative
        chain rather than a defect: exact identity at init means some factor is
        zero, and whatever sits upstream of it waits one step. The pathway is
        fully live from the second step onwards.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        embedding_specs: Optional[Dict[str, Any]],
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.specs = dict(embedding_specs.items())
        self.out_dim = out_dim or base_dim

        # build one embedder per feature
        self.embedders = nn.ModuleDict()
        for name, spec in self.specs.items():
            E = spec["emb_dim"]
            use_bias = spec.get("use_bias", True)
            if spec["type"] == "categorical":
                self.embedders[name] = nn.Embedding(spec["num_classes"], E)
            elif spec["type"] == "continuous":
                self.embedders[name] = nn.Sequential(
                    nn.Linear(spec["in_dim"], E, bias=use_bias),
                    nn.SiLU(),
                    nn.Linear(E, E, bias=use_bias),
                )
            else:
                raise ValueError(f"Unknown type {spec['type']} for feature {name}")
            if spec.get("zero_init", False):
                _zero_last_layer(self.embedders[name])

        # build the single concat→SiLU→Linear head
        total_dim = sum(spec["emb_dim"] for spec in self.specs.values())
        self.project = nn.Sequential(
            nn.Linear(total_dim, self.out_dim, bias=False),
            nn.SiLU(),
        )

    def forward(
        self,
        batch: torch.Tensor,  # [N_nodes,] graph indices
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        features[name] is either [N_graphs, …] or [N_nodes, …]
        and we upsample any per‐graph ones via feat[batch].
        Returns: [N_nodes, out_dim]
        """
        embs = []
        for name, spec in self.specs.items():
            feat = features[name]
            if spec["per"] == "graph":
                feat = feat[batch].unsqueeze(-1)  # now [N_nodes, …]
            if spec["type"] == "categorical":
                feat = (feat + spec.get("offset", 0)).long().squeeze(-1)  # [N_nodes, 1]
            elif spec["type"] == "continuous":
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                scale = spec.get("scale", 1.0)
                if scale != 1.0:
                    feat = feat * scale
            emb = self.embedders[name](feat)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self.project(x)
