import os

def _env_on(name: str, default: str = "0") -> bool:
    """Return True if env var is truthy."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def apply_torchao_w8a16(model):
    """
    TorchAO W8A16 (int8 weight-only) quantization.

    Control switches:
      - USE_QUANT: preferred master switch (0 disables, 1 enables)
      - USE_TORCHAO: legacy switch (kept for backward compatibility)
        Priority: USE_QUANT > USE_TORCHAO
    """
    # Master switch: USE_QUANT (preferred)
    if "USE_QUANT" in os.environ:
        enabled = _env_on("USE_QUANT", "0")
        if not enabled:
            print("[TORCHAO] disabled by env (USE_QUANT=0)")
            return model
    else:
        # Legacy switch: USE_TORCHAO (original behavior, default enabled)
        if not _env_on("USE_TORCHAO", "1"):
            print("[TORCHAO] disabled by env (USE_TORCHAO!=1)")
            return model

    from torchao.quantization import quantize_, int8_weight_only
    model.eval()
    quantize_(model, int8_weight_only())
    return model
