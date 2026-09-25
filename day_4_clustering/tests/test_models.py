"""The vendored neural networks — and the published checkpoints they must load.

Needs the optional ``torch`` extra (``uv sync --extra torch``); the rest of the
package must stay importable without it, which the first test pins down. The
checkpoint test is the one that matters for the released bundle: every ``.pt``
in ``data/embeddings/`` must load *strictly* into the architecture it belongs to,
so a student with the downloaded artifacts can actually use them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="optional extra: uv sync --extra torch")

from cluster.models import (  # noqa: E402
    CnnLstmAttention,
    ConvPoolRegressor,
    MaskedSpectralAE,
)
from cluster.models.masked_spectral_ae import make_block_mask  # noqa: E402

N_PIX = 8575
EMBEDDINGS = Path("data/embeddings")

#: state-dict key prefixes -> (class, constructor kwargs, human name)
ARCHITECTURES: dict[tuple[str, ...], tuple[type, dict[str, Any], str]] = {
    ("dec_fc", "decoder", "encoder", "fc"): (
        MaskedSpectralAE,
        {"n_features": N_PIX, "latent_dim": 256},
        "masked autoencoder",
    ),
    ("attention_w", "conv", "dense_0", "dense_1", "lstm", "output"): (
        CnnLstmAttention,
        {"n_features": N_PIX},
        "supervised CNN-LSTM-attention",
    ),
    ("conv", "output"): (
        ConvPoolRegressor,
        {"n_features": N_PIX, "cnn_filters": [1024, 512, 256, 128, 64]},
        "conv-pool regressor",
    ),
}


def _state_dict(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def test_models_package_is_lazy() -> None:
    """`import cluster.models` must not pull torch in, but attributes resolve."""
    import importlib

    module = importlib.import_module("cluster.models")
    assert module.MaskedSpectralAE is MaskedSpectralAE
    assert module.CnnLstmAttention is CnnLstmAttention
    with pytest.raises(AttributeError):
        module.does_not_exist


def test_block_mask_marks_contiguous_blocks() -> None:
    mask = make_block_mask(2, 1000, 0.5, 200, generator=torch.Generator().manual_seed(0))
    assert mask.shape == (2, 1, 1000)
    assert mask.dtype == torch.bool
    assert mask.any(), "a 50% mask must hide something"


def test_masked_ae_forward_and_embed() -> None:
    model = MaskedSpectralAE(n_features=1000, latent_dim=16, cnn_filters=(8, 4))
    out = model(torch.zeros(3, 1, 1000))
    assert out["latent"].shape == (3, 16)
    assert out["recon"].shape == (3, 1, 1000), "reconstruction is cropped back to n_features"
    assert out["mask"].shape == (3, 1, 1000)

    model.eval()
    x = torch.rand(2, 1, 1000)
    with torch.no_grad():
        assert torch.allclose(model.embed(x), model.embed(x)), "embed must not sample a mask"


def test_supervised_model_forward_taps() -> None:
    model = CnnLstmAttention(
        n_features=256, cnn_filters=[8, 4], lstm_units=8, dense_units=(4, 2), output_dim=9
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 256))
    assert out["attention"].shape == (2, 8)
    assert out["output"].shape == (2, 9)


def test_every_published_checkpoint_loads() -> None:
    """Each downloaded `.pt` must match a vendored architecture exactly."""
    checkpoints = sorted(EMBEDDINGS.glob("*.pt"))
    if not checkpoints:
        pytest.skip(f"no checkpoints in {EMBEDDINGS} (run: cluster download --assets)")

    failures: list[str] = []
    for path in checkpoints:
        state = _state_dict(path)
        prefixes = tuple(sorted({key.split(".")[0] for key in state}))
        entry = ARCHITECTURES.get(prefixes)
        if entry is None:
            failures.append(f"{path.name}: unknown architecture {prefixes}")
            continue
        cls, kwargs, name = entry
        model = cls(**kwargs)
        try:
            model.load_state_dict(state)          # strict: no missing/unexpected keys
        except RuntimeError as exc:
            failures.append(f"{path.name} ({name}): {str(exc).splitlines()[0]}")
    assert not failures, "checkpoints do not load into the vendored models:\n" + "\n".join(failures)
