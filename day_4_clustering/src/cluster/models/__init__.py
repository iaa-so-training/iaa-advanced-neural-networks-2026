"""Neural network definitions used by the re-embedding / figure scripts.

Vendored copies (MIT, © 2024 Rafael Dias) of the upstream ``lightsurf`` models,
so the published checkpoints load without that private package:

* :mod:`cluster.models.masked_spectral_ae` — ``MaskedSpectralAE`` (the
  abundance-free masked autoencoder; checkpoints ``masked_ae*.pt``)
* :mod:`cluster.models.deep_models` — ``CnnLstmAttention`` / ``ConvPoolRegressor``
  (the supervised regressors; checkpoints ``model*.pt``)

``torch`` is an optional extra (``uv sync --extra torch``), so nothing here is
imported eagerly — attributes resolve on first access (PEP 562):

    from cluster.models import MaskedSpectralAE
    from cluster.models.deep_models import CnnLstmAttention
"""

from __future__ import annotations

from typing import Any

_LAZY: dict[str, str] = {
    "MaskedSpectralAE": "masked_spectral_ae",
    "make_block_mask": "masked_spectral_ae",
    "CnnLstmAttention": "deep_models",
    "ConvPoolRegressor": "deep_models",
    "LSTMRegressor": "deep_models",
}

__all__ = sorted(_LAZY)


def __getattr__(name: Any) -> Any:  # keeps `import cluster` torch-free
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)
