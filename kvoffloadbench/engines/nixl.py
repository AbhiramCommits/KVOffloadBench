"""NVIDIA Dynamo/NIXL connector path.

The NIXL connector is selected only when both vLLM and the ``nixl`` Python
package are importable. Otherwise construction raises ``EngineUnavailable``
with the reason, and the factory logs the degradation instead of crashing.
"""

from __future__ import annotations

import importlib.util
import logging

from kvoffloadbench.config import EngineConfig, TierConfig
from kvoffloadbench.engines.base import EngineUnavailable
from kvoffloadbench.engines.vllm_lmcache import VLLMEngine

log = logging.getLogger(__name__)


class NIXLEngine(VLLMEngine):
    mode = "real"
    name = "nixl"

    def __init__(
        self,
        cfg: EngineConfig,
        tiers: list[TierConfig],
        run_name: str = "run",
    ) -> None:
        if importlib.util.find_spec("nixl") is None:
            raise EngineUnavailable(
                "NIXL connector requested but the 'nixl' package is not installed "
                "(install nvidia-nixl / vllm with NIXL support)"
            )
        if importlib.util.find_spec("vllm") is None:
            raise EngineUnavailable("NIXL connector requires vLLM, which is not installed")
        super().__init__(
            cfg=cfg,
            tiers=tiers,
            run_name=run_name,
            connector="NixlConnector",
        )
        self.selection_reason = (
            "CUDA + vLLM + nixl available; connector=NixlConnector "
            f"(agent_config={cfg.nixl_agent_config or '{}'})"
        )
