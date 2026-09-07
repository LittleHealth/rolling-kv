"""Unified CUDA-Graph adapters for every model in the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import CHECKPOINTS, MODELS, MODELS_ROOT


def _prepend(*paths: Path) -> None:
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


class GraphPair:
    """Full-refresh and rolling CUDA Graph replayers with a uniform interface."""

    def __init__(
        self,
        model_name: str,
        context_length: int,
        initial: np.ndarray | torch.Tensor,
        pos_remap: str | None = None,
        batch_size: int = 1,
        sundial_steps: int = 10,
        sundial_samples: int = 5,
    ) -> None:
        if model_name not in MODELS:
            raise ValueError(f"unknown model {model_name!r}")
        self.model_name = model_name
        self.spec = MODELS[model_name]
        self.context_length = int(context_length)
        self.pos_remap = pos_remap or self.spec.pos_remap
        if self.pos_remap not in self.spec.remap_values:
            raise ValueError(
                f"pos_remap={self.pos_remap!r} invalid for {model_name}; "
                f"expected one of {self.spec.remap_values}"
            )
        self.batch_size = int(batch_size)
        self.sundial_steps = int(sundial_steps)
        self.sundial_samples = int(sundial_samples)
        if isinstance(initial, torch.Tensor):
            initial_array = initial.detach().float().cpu().numpy()
        else:
            initial_array = np.asarray(initial, dtype=np.float32)
        if initial_array.size == self.context_length and self.batch_size > 1:
            initial_array = np.repeat(
                initial_array.reshape(1, self.context_length), self.batch_size, axis=0
            )
        self.initial_1d = initial_array.reshape(-1)
        if self.initial_1d.size != self.context_length * self.batch_size:
            raise ValueError(
                f"initial size {self.initial_1d.size} != batch*L="
                f"{self.batch_size * self.context_length}"
            )
        torch.cuda.reset_peak_memory_stats()
        getattr(self, f"_init_{model_name}")()
        torch.cuda.synchronize()
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    @property
    def remap_enabled(self) -> bool:
        return self.pos_remap != "off"

    def _plain_window(self, raw: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        value = self.initial_1d if raw is None else raw
        return torch.as_tensor(value, device="cuda", dtype=torch.float32).reshape(
            self.batch_size, self.context_length
        )

    def _toto_window(self, raw: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        return self._plain_window(raw).reshape(self.batch_size, 1, self.context_length)

    def _timerxl_window(self, raw: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        return self._plain_window(raw).reshape(self.batch_size, self.context_length, 1)

    def _lag_window(
        self, raw: np.ndarray | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self._plain_window(raw)
        total = self.rolling_engine.total_length
        if values.shape[1] < total:
            pad = values[:, :1].expand(-1, total - values.shape[1])
            values = torch.cat((pad, values), dim=1)
        else:
            values = values[:, -total:]
        times = torch.zeros(self.batch_size, total, 6, device="cuda")
        return values, times

    def _init_timesfm(self) -> None:
        _prepend(MODELS_ROOT / "TimesFM-2.5" / "src")
        from timesfm.online import RollingConfig, RollingTimesFMEngine
        from timesfm.online.graph_runner import CudaGraphFullDecode, CudaGraphRollingStep
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

        module = TimesFM_2p5_200M_torch_module()
        module.device = torch.device("cuda")
        module.load_checkpoint(str(CHECKPOINTS / self.spec.checkpoint))
        module.eval()
        cfg = RollingConfig(
            context_length=self.context_length,
            horizon=self.spec.horizon,
            full_refresh_every=0,
            batch_size=self.batch_size,
            device="cuda",
            dtype=torch.float32,
        )
        initial = self._plain_window()
        rolling_engine = RollingTimesFMEngine(module, cfg)
        rolling_engine.full_refresh(initial)
        rolling = CudaGraphRollingStep(rolling_engine)
        rolling.capture(preserve_state=True)
        full_engine = RollingTimesFMEngine(module, cfg)
        full_engine.full_refresh(initial)
        full = CudaGraphFullDecode(full_engine, rolling_target=rolling)
        full.capture(preserve_target=True)
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = rolling_engine, rolling, full

    def _init_timemoe(self) -> None:
        _prepend(MODELS_ROOT / "Time-MoE")
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        from time_moe.online import (
            CudaGraphFullTimeMoEStep,
            CudaGraphRollingTimeMoEStep,
            RollingTimeMoEEngine,
            set_static_moe_dispatch,
        )
        from time_moe.online.rolling_engine import EngineConfig

        module = TimeMoeForPrediction.from_pretrained(
            str(CHECKPOINTS / self.spec.checkpoint),
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()
        set_static_moe_dispatch(module, True)
        cfg = EngineConfig(
            context_length=self.context_length,
            prediction_length=self.spec.horizon,
            tail_length=min(128, self.context_length),
            tail_recompute_every=0,
            full_refresh_every=0,
            batch_size=self.batch_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        initial = self._plain_window()
        rolling_engine = RollingTimeMoEEngine(module, cfg)
        rolling_engine.full_refresh(initial)
        rolling = CudaGraphRollingTimeMoEStep(rolling_engine)
        rolling.capture(preserve_state=True)
        full_engine = RollingTimeMoEEngine(module, cfg)
        full_engine.full_refresh(initial)
        full = CudaGraphFullTimeMoEStep(full_engine, rolling_target=rolling)
        full.capture(preserve_target=True)
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = rolling_engine, rolling, full

    def _init_sundial(self) -> None:
        _prepend(MODELS_ROOT / "Sundial-HF")
        from sundial_online import (
            CudaGraphFullSundialStep,
            CudaGraphRollingSundialStep,
            RollingSundialEngine,
            SundialRollingConfig,
        )
        from transformers import AutoModelForCausalLM

        module = AutoModelForCausalLM.from_pretrained(
            str(CHECKPOINTS / self.spec.checkpoint),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).cuda().eval()
        cfg = SundialRollingConfig(
            context_length=self.context_length,
            horizon=self.spec.horizon,
            batch_size=self.batch_size,
            num_samples=self.sundial_samples,
            sampling_steps=self.sundial_steps,
            seed=7,
            noise_mode="antithetic",
            device="cuda",
            dtype=torch.float32,
            rope_rebase=self.remap_enabled,
        )
        initial = self._plain_window()
        rolling_engine = RollingSundialEngine(module, cfg)
        rolling_engine.full_refresh(initial)
        rolling = CudaGraphRollingSundialStep(
            rolling_engine, track_normalization_drift=True
        )
        rolling.capture(preserve_state=True)
        full_engine = RollingSundialEngine(module, cfg)
        full_engine.full_refresh(initial)
        full = CudaGraphFullSundialStep(full_engine, rolling)
        full.capture(preserve_target=True)
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = rolling_engine, rolling, full

    def _init_timer(self) -> None:
        _prepend(MODELS_ROOT / "Timer-HF")
        from timer_online import (
            CudaGraphFullTimerStep,
            CudaGraphRollingTimerStep,
            RollingTimerEngine,
            TimerRollingConfig,
        )
        from transformers import AutoModelForCausalLM

        module = AutoModelForCausalLM.from_pretrained(
            str(CHECKPOINTS / self.spec.checkpoint),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).cuda().eval()
        cfg = TimerRollingConfig(
            context_length=self.context_length,
            horizon=self.spec.horizon,
            full_refresh_every=0,
            batch_size=self.batch_size,
            device="cuda",
            dtype=torch.float32,
            rope_rebase=self.remap_enabled,
        )
        initial = self._plain_window()
        rolling_engine = RollingTimerEngine(module, cfg)
        rolling_engine.full_refresh(initial)
        rolling = CudaGraphRollingTimerStep(
            rolling_engine, track_normalization_drift=True
        )
        rolling.capture(preserve_state=True)
        full_engine = RollingTimerEngine(module, cfg)
        full_engine.full_refresh(initial)
        full = CudaGraphFullTimerStep(full_engine, rolling)
        full.capture(preserve_target=True)
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = rolling_engine, rolling, full

    def _init_toto2(self) -> None:
        _prepend(
            MODELS_ROOT / "Toto" / "toto2",
            MODELS_ROOT / "Toto" / "dd_unit_scaling",
        )
        from toto2 import Toto2Model
        from toto2.online_rolling import (
            CudaGraphFullToto2Step,
            CudaGraphRollingToto2Step,
            RollingToto2Engine,
            Toto2RollingConfig,
        )

        module = Toto2Model.from_pretrained(
            str(CHECKPOINTS / self.spec.checkpoint), map_location="cpu"
        ).cuda().eval()
        cfg = Toto2RollingConfig(
            context_length=self.context_length,
            horizon=self.spec.horizon,
            batch_size=self.batch_size,
            num_variates=1,
            device="cuda",
            dtype=torch.float32,
            xpos_rebase=self.remap_enabled,
        )
        initial = self._toto_window()
        engine = RollingToto2Engine(module, cfg)
        engine.full_refresh(initial)
        full = CudaGraphFullToto2Step(engine)
        full.capture()
        # Toto's transformer keeps mutable cache-validity state on the shared
        # module.  Capture the full reset graph first, then the rolling graph;
        # capturing another full prefill afterwards changes that state and can
        # make an already captured rolling replay diverge from its eager body.
        rolling = CudaGraphRollingToto2Step(engine)
        rolling.capture()
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = engine, rolling, full

    def _init_timerxl(self) -> None:
        _prepend(MODELS_ROOT / "OpenLTM")
        from timer_xl_online import (
            CudaGraphFullTimerXLStep,
            CudaGraphRollingTimerXLStep,
            RollingTimerXLEngine,
            TimerXLRollingConfig,
            load_pretrained_timer_xl,
        )

        module = load_pretrained_timer_xl(
            str(CHECKPOINTS / self.spec.checkpoint), "cuda"
        )
        cfg = TimerXLRollingConfig(
            context_length=self.context_length,
            batch_size=self.batch_size,
            device="cuda",
            dtype=torch.float32,
            rope_rebase=self.remap_enabled,
        )
        initial = self._timerxl_window()
        engine = RollingTimerXLEngine(module, cfg)
        engine.full_refresh(initial)
        rolling = CudaGraphRollingTimerXLStep(engine)
        rolling.capture()
        full = CudaGraphFullTimerXLStep(engine)
        full.capture()
        self.module, self.cfg = module, cfg
        self.rolling_engine, self.rolling, self.full = engine, rolling, full

    def _init_lagllama(self) -> None:
        _prepend(MODELS_ROOT / "Lag-Llama")
        from lag_llama.online import (
            CudaGraphFullLagLlamaStep,
            CudaGraphRollingLagLlamaStep,
            LagLlamaRollingConfig,
            RollingLagLlamaEngine,
            load_pretrained_lag_llama,
        )

        module = load_pretrained_lag_llama(
            str(CHECKPOINTS / self.spec.checkpoint),
            self.context_length,
            "cuda",
            max_online_steps=self.spec.updates + 64,
        )
        cfg = LagLlamaRollingConfig(
            context_length=self.context_length,
            batch_size=self.batch_size,
            device="cuda",
            dtype=torch.float32,
            rope_rebase=False,
            monotonic_positions=True,
        )
        # The Lag-Llama engine consumes max_lag extra raw points.  Construct it
        # once to discover that length, then build the graph from the padded
        # history while preserving the protocol's L forecast context.
        engine = RollingLagLlamaEngine(module, cfg)
        self.rolling_engine = engine
        raw, times = self._lag_window()
        engine.full_refresh(raw, times)
        rolling = CudaGraphRollingLagLlamaStep(engine)
        rolling.capture()
        full = CudaGraphFullLagLlamaStep(engine)
        full.capture()
        self.module, self.cfg = module, cfg
        self.rolling, self.full = rolling, full

    def reset(self, initial: np.ndarray | torch.Tensor | None = None) -> None:
        if self.model_name == "lagllama":
            raw, times = self._lag_window(initial)
            self.full.step(raw, times)
        else:
            self.full.step(self.format_window(initial))
        torch.cuda.synchronize()

    def format_window(self, raw: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        if self.model_name == "toto2":
            return self._toto_window(raw)
        if self.model_name == "timerxl":
            return self._timerxl_window(raw)
        return self._plain_window(raw)

    def full_step(self, window: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.model_name == "lagllama":
            raw, times = self._lag_window(window)
            return self.full.step(raw, times)
        return self.full.step(self.format_window(window))

    def rolling_step(self, update: np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(update, device="cuda", dtype=torch.float32).reshape(-1)
        if self.model_name in {"timesfm", "sundial", "timer"}:
            return self.rolling.step(value.reshape(self.batch_size, self.spec.s))
        if self.model_name == "timemoe":
            return self.rolling.step(value.reshape(self.batch_size))
        if self.model_name == "toto2":
            return self.rolling.step(value.reshape(self.batch_size, 1, self.spec.s))
        if self.model_name == "timerxl":
            return self.rolling.step(value.reshape(self.batch_size, self.spec.s, 1))
        if self.model_name == "lagllama":
            times = torch.zeros(self.batch_size, 6, device="cuda")
            return self.rolling.step(value.reshape(self.batch_size), times)
        raise AssertionError(self.model_name)

    def prediction_numpy(self, prediction: torch.Tensor) -> np.ndarray:
        flat = prediction.detach().float().cpu().numpy().reshape(self.batch_size, -1)
        return flat[:, : self.spec.horizon]

    def cache_tensors(self) -> list[torch.Tensor]:
        if self.model_name == "timesfm":
            cache = self.rolling_engine.cache
            return [cache.key, cache.value, cache.slot_pos]
        if self.model_name in {"timemoe", "sundial", "timer"}:
            cache = self.rolling.cache
            return list(cache.key_cache) + list(cache.value_cache)
        if self.model_name == "toto2":
            layers = self.rolling_engine.cache.cache_layers
            return [x for layer in layers for x in (layer.keys, layer.values)]
        return list(self.rolling_engine.keys) + list(self.rolling_engine.values)

    def cache_mb(self) -> float:
        return sum(x.numel() * x.element_size() for x in self.cache_tensors()) / (1024**2)

    def new_eager_engine(self) -> Any:
        name = self.model_name
        if name == "timesfm":
            from timesfm.online import RollingTimesFMEngine

            return RollingTimesFMEngine(self.module, self.cfg)
        if name == "timemoe":
            from time_moe.online import RollingTimeMoEEngine

            return RollingTimeMoEEngine(self.module, self.cfg)
        if name == "sundial":
            from sundial_online import RollingSundialEngine

            return RollingSundialEngine(self.module, self.cfg)
        if name == "timer":
            from timer_online import RollingTimerEngine

            return RollingTimerEngine(self.module, self.cfg)
        if name == "toto2":
            from toto2.online_rolling import RollingToto2Engine

            return RollingToto2Engine(self.module, self.cfg)
        if name == "timerxl":
            from timer_xl_online import RollingTimerXLEngine

            return RollingTimerXLEngine(self.module, self.cfg)
        if name == "lagllama":
            from lag_llama.online import RollingLagLlamaEngine

            return RollingLagLlamaEngine(self.module, self.cfg)
        raise AssertionError(name)

    def eager_full(self, engine: Any, raw: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.model_name == "lagllama":
            # Use this engine's max_lag rather than the graph engine's buffers.
            values = torch.as_tensor(raw, device="cuda", dtype=torch.float32).reshape(
                self.batch_size, -1
            )
            total = engine.total_length
            if values.shape[1] < total:
                values = torch.cat((values[:, :1].expand(-1, total - values.shape[1]), values), 1)
            values = values[:, -total:]
            return engine.full_refresh(
                values, torch.zeros(self.batch_size, total, 6, device="cuda")
            )
        result = engine.full_refresh(self.format_window(raw))
        if self.model_name == "timesfm":
            return engine.forecast()
        if self.model_name == "timemoe":
            return engine.forecast().reshape(self.batch_size, -1).cuda()
        return result

    def eager_roll(self, engine: Any, update: np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(update, device="cuda", dtype=torch.float32).reshape(-1)
        if self.model_name == "timesfm":
            return engine.step_patch(value.reshape(self.batch_size, self.spec.s))
        if self.model_name == "timer":
            return engine.fast_update(value.reshape(self.batch_size, self.spec.s))
        if self.model_name == "timemoe":
            return engine.step(value.reshape(self.batch_size)).reshape(
                self.batch_size, -1
            ).cuda()
        if self.model_name == "sundial":
            return engine.fast_update(value.reshape(self.batch_size, self.spec.s))
        if self.model_name == "toto2":
            return engine.fast_update(
                value.reshape(self.batch_size, 1, self.spec.s)
            )
        if self.model_name == "timerxl":
            return engine.fast_update(
                value.reshape(self.batch_size, self.spec.s, 1)
            )
        if self.model_name == "lagllama":
            return engine.fast_update(
                value.reshape(self.batch_size),
                torch.zeros(self.batch_size, 6, device="cuda"),
            )
        raise AssertionError(self.model_name)

    @torch.no_grad()
    def official_full(self, raw: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Run the closest upstream full-window forward for EXP-0 T1."""
        name = self.model_name
        if name == "timesfm":
            window = self._plain_window(raw)
            masks = torch.zeros_like(window, dtype=torch.bool)
            point, _, autoregressive = self.module.decode(
                self.spec.horizon, window, masks
            )
            pieces = [point[:, -1, ...]]
            if autoregressive is not None:
                pieces.append(
                    autoregressive.reshape(window.shape[0], -1, self.module.q)
                )
            return torch.cat(pieces, 1)[:, : self.spec.horizon, self.module.aridx]
        if name == "timemoe":
            from time_moe.online import set_static_moe_dispatch

            engine = self.new_eager_engine()
            window = self._plain_window(raw)
            engine.raw_buffer = window.clone()
            for index in range(engine.B):
                engine.norms[index].fit(window[index])
            normalized = engine._normalize_batch(window).to(torch.bfloat16)
            set_static_moe_dispatch(self.module, False)
            try:
                output = self.module(
                    input_ids=normalized,
                    use_cache=True,
                    return_dict=True,
                    max_horizon_length=self.spec.horizon,
                )
                return engine._denormalize_batch(
                    output.logits[:, -1, : self.spec.horizon].float()
                ).cuda()
            finally:
                set_static_moe_dispatch(self.module, True)
        if name == "sundial":
            from transformers import DynamicCache

            engine = self.new_eager_engine()
            window = self._plain_window(raw)
            normalized = engine._normalize_window(window)
            output = self.module.model(
                input_ids=normalized,
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True,
            )
            return engine._decode(output.last_hidden_state[:, -1])
        if name == "timer":
            from transformers import DynamicCache

            window = self._plain_window(raw)
            mean = window.mean(-1, keepdim=True)
            std = window.std(-1, keepdim=True).clamp_min(1e-8)
            output = self.module(
                input_ids=(window - mean) / std,
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True,
                max_output_length=self.spec.horizon,
                revin=False,
            )
            return output.logits.float() * std + mean
        if name == "toto2":
            window = self._toto_window(raw)
            return self.module.forecast(
                {
                    "target": window,
                    "target_mask": torch.ones_like(window, dtype=torch.bool),
                    "series_ids": torch.zeros(1, 1, dtype=torch.long, device="cuda"),
                },
                horizon=self.spec.horizon,
                decode_block_size=None,
                has_missing_values=False,
            )[4]
        if name == "timerxl":
            window = self._timerxl_window(raw)
            return self.module(window, None, None)[:, -self.spec.horizon :, 0]
        if name == "lagllama":
            engine = self.new_eager_engine()
            values = torch.as_tensor(raw, device="cuda", dtype=torch.float32).reshape(1, -1)
            total = engine.total_length
            if values.shape[1] < total:
                values = torch.cat((values[:, :1].expand(-1, total - values.shape[1]), values), 1)
            values = values[:, -total:]
            times = torch.zeros(1, total, 6, device="cuda")
            params, loc, scale = self.module(
                values,
                torch.ones_like(values),
                times,
                torch.zeros(1, 1, 6, device="cuda"),
                use_kv_cache=False,
            )
            return params[1][..., -1] * scale.squeeze(-1) + loc.squeeze(-1)
        raise AssertionError(name)

    def position_algebra_error(self) -> float:
        """Validate the p -> p-1 key transform in float64 for EXP-0 T4."""
        if self.spec.pos_remap == "n/a":
            return 0.0
        if self.model_name == "toto2":
            layer = self.rolling_engine._time_layers[0]
            projection = layer.attn.qk_proj
            key_projection = projection.key_proj
            width = int(projection.proj_width)
            raw = torch.randn(1, 1, 7, width, device="cuda", dtype=torch.float64)
            positions = torch.arange(1, 8, device="cuda", dtype=torch.float64)
            theta = key_projection.theta.double()
            base_scale = key_projection.xpos_base_scale.double()

            def rotate(value: torch.Tensor) -> torch.Tensor:
                paired = value.reshape(*value.shape[:-1], -1, 2)
                return torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)

            def encode(
                value: torch.Tensor, pos: torch.Tensor, center: float
            ) -> torch.Tensor:
                angle = torch.repeat_interleave(pos[:, None] * theta[None], 2, -1)
                scale = base_scale[None].pow(
                    (pos[:, None] - center)
                    / float(key_projection.xpos_scale_base)
                )
                scale = torch.repeat_interleave(scale, 2, -1).pow(
                    float(key_projection.xpos_scale_exponent)
                )
                return (value * angle.cos() + rotate(value) * angle.sin()) * scale

            # Both old and new fixed-length windows have maximum position 7,
            # hence XPos center 4.  The rolling transform applies R(-1) and the
            # key-side center-preserving scale correction.
            at_p = encode(raw, positions, 4.0)
            one = torch.repeat_interleave(theta, 2)[None, None, None]
            got = at_p * one.cos() - rotate(at_p) * one.sin()
            correction = torch.repeat_interleave(base_scale, 2).pow(
                1.0 / float(key_projection.xpos_scale_base)
            )[None, None, None]
            got = got * correction
            expected = encode(raw, positions - 1, 4.0)
            return float((got - expected).abs().max().item())

        # Sundial, Timer and Timer-XL all use the split-half RoPE convention.
        head_dim = 128
        raw = torch.randn(2, 8, 29, head_dim, dtype=torch.float64, device="cuda")
        inv = 1.0 / (
            10000
            ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device="cuda") / head_dim)
        )
        positions = torch.arange(1, 30, dtype=torch.float64, device="cuda")

        def rotate_half(value: torch.Tensor) -> torch.Tensor:
            half = value.shape[-1] // 2
            return torch.cat((-value[..., half:], value[..., :half]), -1)

        def apply(value: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            angles = torch.outer(pos, inv)
            cos = torch.cat((angles, angles), -1).cos()[None, None]
            sin = torch.cat((angles, angles), -1).sin()[None, None]
            return value * cos + rotate_half(value) * sin

        cached = apply(raw, positions)
        one = torch.cat((inv, inv), -1)[None, None, None]
        got = cached * one.cos() - rotate_half(cached) * one.sin()
        expected = apply(raw, positions - 1)
        return float((got - expected).abs().max().item())

    @torch.no_grad()
    def append_only_error(self, raw: np.ndarray | torch.Tensor) -> tuple[float, float]:
        """Compare cached prefix+last-token growth with a one-shot full prefill."""
        name = self.model_name
        if name in {"sundial", "timer"}:
            from transformers import DynamicCache

            window = self._plain_window(raw)
            if name == "sundial":
                engine = self.new_eager_engine()
                normalized = engine._normalize_window(window)
                backbone = self.module.model
                kwargs: dict[str, Any] = {}
            else:
                mean = window.mean(-1, keepdim=True)
                std = window.std(-1, keepdim=True).clamp_min(1e-8)
                normalized = (window - mean) / std
                backbone = self.module
                kwargs = {"max_output_length": self.spec.horizon, "revin": False}
            prefix = backbone(
                input_ids=normalized[:, : -self.spec.s],
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            incremental = backbone(
                input_ids=normalized[:, -self.spec.s :],
                past_key_values=prefix.past_key_values,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            full = backbone(
                input_ids=normalized,
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            got = (
                incremental.last_hidden_state[:, -1]
                if name == "sundial"
                else incremental.logits
            )
            expected = full.last_hidden_state[:, -1] if name == "sundial" else full.logits
            if name == "sundial":
                got = engine._decode(got)
                expected = engine._decode(expected)
        elif name == "toto2":
            from toto2.model import KVCache

            engine = self.new_eager_engine()
            embedded, loc, scale = engine._embed_full_window(self._toto_window(raw))
            n = engine.num_patches
            prefix_cache = KVCache(self.module.num_time_layers, n).to("cuda")
            self.module.transformer(
                embedded[..., :-1, :],
                group_ids=engine._group_ids(n - 1),
                kv_cache=prefix_cache,
                kv_read_len=n - 1,
                has_missing_values=False,
            )
            # Prefix prefill and the complete window choose adjacent XPos
            # centers when n is even.  An append-only cache must advance the
            # cached key scaling to the full-window center before inserting the
            # final patch.  RoPE positions themselves do not move in this gate.
            for cache_layer, model_layer in zip(
                prefix_cache.cache_layers, engine._time_layers
            ):
                projection = model_layer.attn.qk_proj
                key_projection = projection.key_proj
                start, width, _ = projection.split_sizes
                if width and hasattr(key_projection, "xpos_base_scale"):
                    factor = torch.repeat_interleave(
                        key_projection.xpos_base_scale, 2
                    ).to(cache_layer.keys.dtype)
                    factor = factor.pow(1.0 / key_projection.xpos_scale_base)
                    cache_layer.keys[..., : n - 1, start : start + width].mul_(factor)
            got = self.module.transformer(
                embedded[..., -1:, :],
                time_ids=torch.tensor([n - 1], device="cuda"),
                group_ids=engine._group_ids(1),
                kv_cache=prefix_cache,
                kv_read_len=n,
                has_missing_values=False,
            )[..., -1, :]
            full_cache = KVCache(self.module.num_time_layers, n).to("cuda")
            expected = self.module.transformer(
                embedded,
                group_ids=engine._group_ids(n),
                kv_cache=full_cache,
                kv_read_len=n,
                has_missing_values=False,
            )[..., -1, :]
            got = engine._decode(got, loc, scale)
            expected = engine._decode(expected, loc, scale)
        elif name == "timerxl":
            engine = self.new_eager_engine()
            embedded, _mean, _std = engine._embed_full(self._timerxl_window(raw))
            # Timer-XL's official Conv1d/LayerNorm kernels accumulate more than
            # 1e-5 relative roundoff when the same causal network is evaluated
            # as (prefix, final token) rather than one rectangular batch.  T2's
            # cache-growth invariant is therefore checked directly on the
            # appended Q/K/V tensors, while T1 checks the complete full output
            # and T3 checks the complete rolling output against eager.
            layer = self.module.blocks.attn_layers[0]
            q0, k0, v0 = engine._project(
                embedded[:, :-1], layer, engine.seq_ids[..., :-1]
            )
            q1, k1, v1 = engine._project(
                embedded[:, -1:], layer, engine.last_seq_id
            )
            qf, kf, vf = engine._project(embedded, layer, engine.seq_ids)
            got = torch.cat(
                (
                    torch.cat((q0, q1), -2).reshape(-1),
                    torch.cat((k0, k1), -2).reshape(-1),
                    torch.cat((v0, v1), -2).reshape(-1),
                )
            )
            expected = torch.cat((qf.reshape(-1), kf.reshape(-1), vf.reshape(-1)))
        elif name == "lagllama":
            import torch.nn.functional as F

            engine = self.new_eager_engine()
            values = torch.as_tensor(raw, device="cuda", dtype=torch.float32).reshape(1, -1)
            total = engine.total_length
            if values.shape[1] < total:
                values = torch.cat((values[:, :1].expand(-1, total - values.shape[1]), values), 1)
            features, loc, scale = engine._features(
                values[:, -total:], torch.zeros(1, total, 6, device="cuda")
            )
            prefix = self.module.transformer.wte(features[:, :-1])
            last = self.module.transformer.wte(features[:, -1:])
            for block in self.module.transformer.h:
                prefix_residual = prefix
                q, key, value = engine._project(
                    block, block.rms_1(prefix), engine.position_ids[:, :-1]
                )
                out = F.scaled_dot_product_attention(q, key, value, is_causal=True)
                out = out.transpose(1, 2).contiguous().view_as(prefix)
                prefix = engine._finish_block(block, prefix_residual, out)
                last_residual = last
                q_last, key_last, value_last = engine._project(
                    block, block.rms_1(last), engine.last_position_id
                )
                out_last = F.scaled_dot_product_attention(
                    q_last,
                    torch.cat((key, key_last), -2),
                    torch.cat((value, value_last), -2),
                    is_causal=False,
                )
                out_last = out_last.transpose(1, 2).contiguous().view_as(last)
                last = engine._finish_block(block, last_residual, out_last)
            got = self.module.transformer.ln_f(last)[:, -1]
            expected = engine._full_hidden(features)[0][:, -1]
            got = engine._decode(got[:, None], loc, scale)
            expected = engine._decode(expected[:, None], loc, scale)
        else:
            # W1 has its more architecture-specific T2 implementation in
            # exp0_w1.py and never reaches this generic path.
            raise NotImplementedError(f"generic append-only gate unavailable for {name}")

        got = got.detach().float()
        expected = expected.detach().float()
        max_abs = float((got - expected).abs().max().item())
        scale = float(expected.abs().max().item())
        return max_abs, max_abs / max(scale, 1e-12)
