"""Graph-safe online sliding cache for Lag-Llama one-step forecasting."""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from lag_llama.model.module import LagLlamaModel, rotate_half


def load_pretrained_lag_llama(
    checkpoint, context_length, device="cuda", max_online_steps=8192
):
    document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kwargs = dict(document["hyper_parameters"]["model_kwargs"])
    kwargs["context_length"] = context_length
    kwargs["max_context_length"] = max(
        context_length + max_online_steps, kwargs["max_context_length"]
    )
    model = LagLlamaModel(**kwargs)
    state = {
        key.removeprefix("model."): value
        for key, value in document["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state)
    return model.to(device).eval()


@dataclasses.dataclass
class LagLlamaRollingConfig:
    context_length: int = 512
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    rope_rebase: bool = True
    monotonic_positions: bool = True


class RollingLagLlamaEngine:
    def __init__(self, model: LagLlamaModel, cfg: LagLlamaRollingConfig):
        self.model = model.eval()
        self.cfg = cfg
        self.lags = list(model.lags_seq)
        self.max_lag = max(self.lags)
        self.total_length = self.max_lag + cfg.context_length
        self.raw_buffer = self.time_buffer = self.keys = self.values = None
        self.norm_loc = self.norm_scale = None
        self.last_prediction = None
        positions = torch.arange(cfg.context_length, device=cfg.device, dtype=torch.long)
        self.position_ids = positions.unsqueeze(0).expand(cfg.batch_size, -1)
        self.last_position_id = self.position_ids[:, -1:]
        self.next_position_id = torch.full(
            (cfg.batch_size, 1),
            cfg.context_length,
            device=cfg.device,
            dtype=torch.long,
        )
        token_index = torch.arange(cfg.context_length, device=cfg.device).unsqueeze(1)
        lag_index = torch.tensor(self.lags, device=cfg.device).unsqueeze(0)
        self.lag_indices = self.max_lag + token_index - lag_index
        self.last_lag_indices = self.total_length - 1 - lag_index.squeeze(0)

    def _raw(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (self.cfg.batch_size, self.total_length)
        if tensor.shape != expected:
            raise ValueError(f"expected raw history {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _time(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (self.cfg.batch_size, self.total_length, 6)
        if tensor.shape != expected:
            raise ValueError(f"expected time features {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _stats(self, raw):
        ordered = raw.sort(dim=-1).values
        loc = ordered[..., (self.total_length - 1) // 2 : (self.total_length - 1) // 2 + 1]

        def quantile(q):
            position = q * (self.total_length - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            fraction = position - lower
            return ordered[..., lower : lower + 1] * (1.0 - fraction) + ordered[..., upper : upper + 1] * fraction

        scale = (quantile(0.75) - quantile(0.25)).clamp_min(1e-10)
        return loc, scale

    def _features(self, raw, time_features):
        loc, scale = self._stats(raw)
        normalized = (raw - loc) / scale
        lags = normalized[:, self.lag_indices]
        static = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)
        static = static.unsqueeze(1).expand(-1, self.cfg.context_length, -1)
        features = torch.cat((lags, static, time_features[:, self.max_lag :]), dim=-1)
        return features, loc, scale

    def _last_features(self, raw, time_features, loc=None, scale=None):
        if loc is None or scale is None:
            loc, scale = self._stats(raw)
        normalized = (raw - loc) / scale
        lags = normalized[:, self.last_lag_indices].unsqueeze(1)
        static = torch.cat((loc.abs().log1p(), scale.log()), dim=-1).unsqueeze(1)
        features = torch.cat((lags, static, time_features[:, -1:]), dim=-1)
        return features, loc, scale

    @staticmethod
    def _project(block, x, position_ids):
        B, T, _ = x.shape
        attn = block.attn
        q = attn.q_proj(x)
        k, v = attn.kv_proj(x).chunk(2, dim=-1)
        q = q.view(B, T, attn.n_head, attn.n_embd_per_head).transpose(1, 2)
        k = k.view(B, T, attn.n_head, attn.n_embd_per_head).transpose(1, 2)
        v = v.view(B, T, attn.n_head, attn.n_embd_per_head).transpose(1, 2)
        if attn.rotary_emb is not None:
            cos = attn.rotary_emb.cos_cached[0, 0][position_ids].unsqueeze(1).to(q.dtype)
            sin = attn.rotary_emb.sin_cached[0, 0][position_ids].unsqueeze(1).to(q.dtype)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        return q, k, v

    @staticmethod
    def _finish_block(block, x, attn_out):
        x = x + block.attn.c_proj(attn_out)
        return x + block.mlp(block.rms_2(x))

    def _full_hidden(self, features, store_cache=False):
        x = self.model.transformer.wte(features)
        keys, values = [], []
        for block in self.model.transformer.h:
            residual = x
            q, k, v = self._project(block, block.rms_1(x), self.position_ids)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view_as(x)
            x = self._finish_block(block, residual, out)
            if store_cache:
                keys.append(k.clone())
                values.append(v.clone())
        return self.model.transformer.ln_f(x), keys, values

    @staticmethod
    def _rebase_key_(key, block):
        rotary = block.attn.rotary_emb
        cos = rotary.cos_cached[0, 0, 1].to(key.dtype)
        sin = -rotary.sin_cached[0, 0, 1].to(key.dtype)
        key.copy_(key * cos + rotate_half(key) * sin)

    def _rolling_hidden(self, features):
        x = self.model.transformer.wte(features)
        keep = self.cfg.context_length - 1
        for index, block in enumerate(self.model.transformer.h):
            residual = x
            position_id = (
                self.next_position_id
                if self.cfg.monotonic_positions
                else self.last_position_id
            )
            q, k_new, v_new = self._project(block, block.rms_1(x), position_id)
            key, value = self.keys[index], self.values[index]
            key[..., :keep, :].copy_(key[..., 1:, :].clone())
            value[..., :keep, :].copy_(value[..., 1:, :].clone())
            if self.cfg.rope_rebase and not self.cfg.monotonic_positions:
                self._rebase_key_(key[..., :keep, :], block)
            key[..., keep:, :].copy_(k_new)
            value[..., keep:, :].copy_(v_new)
            out = F.scaled_dot_product_attention(q, key, value, is_causal=False)
            out = out.transpose(1, 2).contiguous().view_as(x)
            x = self._finish_block(block, residual, out)
        if self.cfg.monotonic_positions:
            self.next_position_id.add_(1)
        return self.model.transformer.ln_f(x)

    def _decode(self, hidden, loc, scale):
        params = self.model.param_proj(hidden)
        return params[1][..., -1] * scale.squeeze(-1) + loc.squeeze(-1)

    @torch.no_grad()
    def full_refresh(self, raw_history, time_features):
        self.raw_buffer = self._raw(raw_history).clone()
        self.time_buffer = self._time(time_features).clone()
        features, loc, scale = self._features(self.raw_buffer, self.time_buffer)
        hidden, self.keys, self.values = self._full_hidden(features, store_cache=True)
        self.next_position_id.fill_(self.cfg.context_length)
        self.norm_loc, self.norm_scale = loc, scale
        self.last_prediction = self._decode(hidden[:, -1:], loc, scale)
        return self.last_prediction

    @torch.no_grad()
    def fast_update(self, new_value, new_time_feature):
        value = torch.as_tensor(new_value, device=self.cfg.device, dtype=self.cfg.dtype).view(self.cfg.batch_size, 1)
        time_feature = torch.as_tensor(new_time_feature, device=self.cfg.device, dtype=self.cfg.dtype).view(self.cfg.batch_size, 1, 6)
        self.raw_buffer = torch.cat((self.raw_buffer[:, 1:], value), -1)
        self.time_buffer = torch.cat((self.time_buffer[:, 1:], time_feature), 1)
        features, loc, scale = self._last_features(
            self.raw_buffer, self.time_buffer, self.norm_loc, self.norm_scale
        )
        hidden = self._rolling_hidden(features)
        self.last_prediction = self._decode(hidden, loc, scale)
        return self.last_prediction


class CudaGraphRollingLagLlamaStep:
    def __init__(self, engine: RollingLagLlamaEngine, warmup=3):
        self.engine = engine
        self.warmup = warmup
        self.value = torch.zeros(engine.cfg.batch_size, 1, device=engine.cfg.device)
        self.time_feature = torch.zeros(engine.cfg.batch_size, 1, 6, device=engine.cfg.device)
        self.graph = self.output = None

    def _body(self):
        eng = self.engine
        eng.raw_buffer.copy_(torch.cat((eng.raw_buffer[:, 1:], self.value), -1))
        eng.time_buffer.copy_(torch.cat((eng.time_buffer[:, 1:], self.time_feature), 1))
        features, loc, scale = eng._last_features(
            eng.raw_buffer, eng.time_buffer, eng.norm_loc, eng.norm_scale
        )
        hidden = eng._rolling_hidden(features)
        return eng._decode(hidden, loc, scale)

    @torch.no_grad()
    def capture(self):
        eng = self.engine
        raw, times = eng.raw_buffer.clone(), eng.time_buffer.clone()
        next_position = eng.next_position_id.clone()
        keys, values = [x.clone() for x in eng.keys], [x.clone() for x in eng.values]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._body()
        eng.raw_buffer.copy_(raw)
        eng.time_buffer.copy_(times)
        eng.next_position_id.copy_(next_position)
        for dst, src in zip(eng.keys, keys): dst.copy_(src)
        for dst, src in zip(eng.values, values): dst.copy_(src)

    @torch.no_grad()
    def step(self, value, time_feature):
        self.value.copy_(torch.as_tensor(value, device=self.value.device).view_as(self.value))
        self.time_feature.copy_(torch.as_tensor(time_feature, device=self.time_feature.device).view_as(self.time_feature))
        self.graph.replay()
        return self.output


class CudaGraphFullLagLlamaStep:
    def __init__(self, engine: RollingLagLlamaEngine, warmup=3):
        self.engine = engine
        self.warmup = warmup
        self.raw = engine.raw_buffer.clone()
        self.times = engine.time_buffer.clone()
        self.graph = self.output = None

    def _body(self):
        eng = self.engine
        features, loc, scale = eng._features(self.raw, self.times)
        hidden, keys, values = eng._full_hidden(features, store_cache=True)
        eng.raw_buffer.copy_(self.raw)
        eng.time_buffer.copy_(self.times)
        eng.norm_loc.copy_(loc)
        eng.norm_scale.copy_(scale)
        eng.next_position_id.fill_(eng.cfg.context_length)
        for dst, src in zip(eng.keys, keys):
            dst.copy_(src)
        for dst, src in zip(eng.values, values):
            dst.copy_(src)
        return eng._decode(hidden[:, -1:], loc, scale)

    @torch.no_grad()
    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup): self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.output = self._body()

    @torch.no_grad()
    def step(self, raw, times):
        self.raw.copy_(torch.as_tensor(raw, device=self.raw.device).view_as(self.raw))
        self.times.copy_(torch.as_tensor(times, device=self.times.device).view_as(self.times))
        self.graph.replay()
        return self.output
