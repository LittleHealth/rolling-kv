"""Univariate online rolling cache for the official Timer-XL PyTorch model."""

from __future__ import annotations

import argparse
import dataclasses
import math

import torch
import torch.nn.functional as F

from models.timer_xl import Model


def timer_xl_pretrained_args():
    return argparse.Namespace(
        input_token_len=96,
        output_token_len=96,
        d_model=1024,
        n_heads=8,
        e_layers=8,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        use_norm=True,
        flash_attention=False,
        covariate=False,
        output_attention=False,
    )


def load_pretrained_timer_xl(checkpoint, device="cuda"):
    model = Model(timer_xl_pretrained_args())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


@dataclasses.dataclass
class TimerXLRollingConfig:
    context_length: int = 12288
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    rope_rebase: bool = True


class RollingTimerXLEngine:
    def __init__(self, model: Model, cfg: TimerXLRollingConfig):
        self.model = model.eval()
        self.cfg = cfg
        self.patch = int(model.input_token_len)
        if cfg.context_length <= 0 or cfg.context_length % self.patch:
            raise ValueError(f"context_length must be a multiple of {self.patch}")
        self.num_tokens = cfg.context_length // self.patch
        self.raw_buffer = self.keys = self.values = None
        self.norm_mean = self.norm_std = None
        self.last_prediction = None
        self.seq_ids = torch.arange(
            self.num_tokens, device=cfg.device, dtype=torch.long
        ).view(1, 1, -1).expand(cfg.batch_size, 8, -1)
        self.last_seq_id = self.seq_ids[..., -1:]
        for layer in self.model.blocks.attn_layers:
            layer.attention.inner_attention.qk_proj.query_proj._init_freq(
                self.num_tokens
            )

    def _window(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (self.cfg.batch_size, self.cfg.context_length, 1)
        if tensor.shape != expected:
            raise ValueError(f"expected {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _patch(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (self.cfg.batch_size, self.patch, 1)
        if tensor.shape != expected:
            raise ValueError(f"expected {expected}, got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _stats(window):
        mean = window.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(window, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return mean, std

    def _embed_full(self, window):
        mean, std = self._stats(window)
        normalized = (window - mean) / std
        patches = normalized.permute(0, 2, 1).unfold(-1, self.patch, self.patch)
        embedded = self.model.embedding(patches).reshape(
            self.cfg.batch_size, self.num_tokens, -1
        )
        return embedded, mean, std

    def _rope(self, q, k, seq_ids, attention):
        projection = attention.inner_attention.qk_proj
        start, width, _ = projection.split_sizes
        q_parts = list(q.split(projection.split_sizes, dim=-1))
        k_parts = list(k.split(projection.split_sizes, dim=-1))
        rotary = projection.query_proj
        cos = rotary.cos[seq_ids].to(q.dtype)
        sin = rotary.sin[seq_ids].to(q.dtype)
        q_parts[1] = cos * q_parts[1] + sin * rotary._rotate(q_parts[1])
        k_parts[1] = cos * k_parts[1] + sin * rotary._rotate(k_parts[1])
        return torch.cat(q_parts, -1), torch.cat(k_parts, -1)

    def _project(self, x, layer, seq_ids):
        attention = layer.attention
        heads = attention.n_heads
        q = attention.query_projection(x).view(x.shape[0], x.shape[1], heads, -1)
        k = attention.key_projection(x).view(x.shape[0], x.shape[1], heads, -1)
        v = attention.value_projection(x).view(x.shape[0], x.shape[1], heads, -1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = self._rope(q, k, seq_ids, attention)
        return q, k, v

    @staticmethod
    def _finish_layer(x, attention_out, layer):
        x = x + layer.dropout(attention_out)
        y = x = layer.norm1(x)
        y = layer.dropout(layer.activation(layer.conv1(y.transpose(-1, 1))))
        y = layer.dropout(layer.conv2(y).transpose(-1, 1))
        return layer.norm2(x + y)

    @staticmethod
    def _attention(q, k, v, causal, layer):
        """Match Timer-XL's official non-flash attention arithmetic.

        The released checkpoint has ``flash_attention=False``.  Keeping the
        eager/graph online path on the same einsum + softmax formulation avoids
        backend-dependent drift at the strict EXP-0 FP32 threshold.
        """
        scores = torch.einsum("bhle,bhse->bhls", q, k)
        # With one variate BinaryAttentionBias selects embedding row 1 for
        # every valid pair.  It is mathematically softmax-invariant, but adding
        # it exactly as upstream does is required for bit-level FP32 agreement.
        bias = layer.attention.inner_attention.attn_bias.emb.weight[1]
        scores = scores + bias.view(1, -1, 1, 1)
        if causal:
            future = torch.triu(
                torch.ones(
                    scores.shape[-2:], dtype=torch.bool, device=scores.device
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(future, float("-inf"))
        scores = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        return torch.einsum("bhls,bhsd->blhd", scores, v)

    def _full_hidden(self, embedded, store_cache=False):
        x = embedded
        keys, values = [], []
        for layer in self.model.blocks.attn_layers:
            q, k, v = self._project(x, layer, self.seq_ids)
            out = self._attention(q, k, v, causal=True, layer=layer).reshape(x.shape)
            out = layer.attention.out_projection(out)
            if store_cache:
                keys.append(k.clone())
                values.append(v.clone())
            x = self._finish_layer(x, out, layer)
        x = self.model.blocks.norm(x)
        return x, keys, values

    def _rebase_key_(self, key, layer):
        projection = layer.attention.inner_attention.qk_proj
        start, width, _ = projection.split_sizes
        rotary = projection.query_proj
        rotated = key[..., start : start + width]
        cos = rotary.cos[1].to(rotated.dtype)
        sin = -rotary.sin[1].to(rotated.dtype)
        rotated.copy_(cos * rotated + sin * rotary._rotate(rotated))

    def _rolling_hidden(self, embedded):
        x = embedded
        keep = self.num_tokens - 1
        for index, layer in enumerate(self.model.blocks.attn_layers):
            q, k_new, v_new = self._project(x, layer, self.last_seq_id)
            key = self.keys[index]
            value = self.values[index]
            key[..., :keep, :].copy_(key[..., 1 : self.num_tokens, :].clone())
            value[..., :keep, :].copy_(value[..., 1 : self.num_tokens, :].clone())
            if self.cfg.rope_rebase:
                self._rebase_key_(key[..., :keep, :], layer)
            key[..., keep:, :].copy_(k_new)
            value[..., keep:, :].copy_(v_new)
            out = self._attention(
                q, key, value, causal=False, layer=layer
            ).reshape(x.shape)
            out = layer.attention.out_projection(out)
            x = self._finish_layer(x, out, layer)
        return self.model.blocks.norm(x)

    def _decode(self, hidden, mean, std):
        prediction = self.model.head(hidden)
        return prediction * std[:, 0, :] + mean[:, 0, :]

    @torch.no_grad()
    def full_refresh(self, raw_window):
        window = self._window(raw_window)
        self.raw_buffer = window.clone()
        embedded, mean, std = self._embed_full(window)
        hidden, self.keys, self.values = self._full_hidden(embedded, store_cache=True)
        self.norm_mean, self.norm_std = mean, std
        self.last_prediction = self._decode(hidden[:, -1], mean, std)
        return self.last_prediction

    @torch.no_grad()
    def fast_update(self, new_patch):
        if self.raw_buffer is None:
            raise RuntimeError("call full_refresh first")
        patch = self._patch(new_patch)
        self.raw_buffer = torch.cat((self.raw_buffer[:, self.patch :], patch), dim=1)
        mean, std = self._stats(self.raw_buffer)
        embedded = self.model.embedding(((patch - mean) / std).squeeze(-1)).unsqueeze(1)
        hidden = self._rolling_hidden(embedded)
        self.last_prediction = self._decode(hidden[:, -1], mean, std)
        return self.last_prediction


class CudaGraphRollingTimerXLStep:
    def __init__(self, engine: RollingTimerXLEngine, warmup=3):
        self.engine = engine
        self.warmup = warmup
        self.patch = torch.zeros(
            engine.cfg.batch_size,
            engine.patch,
            1,
            device=engine.cfg.device,
            dtype=engine.cfg.dtype,
        )
        self.graph = self.output = None

    def _body(self):
        eng = self.engine
        eng.raw_buffer.copy_(torch.cat((eng.raw_buffer[:, eng.patch :], self.patch), 1))
        mean, std = eng._stats(eng.raw_buffer)
        embedded = eng.model.embedding(((self.patch - mean) / std).squeeze(-1)).unsqueeze(1)
        hidden = eng._rolling_hidden(embedded)
        return eng._decode(hidden[:, -1], mean, std)

    @torch.no_grad()
    def capture(self):
        eng = self.engine
        raw = eng.raw_buffer.clone()
        keys = [x.clone() for x in eng.keys]
        values = [x.clone() for x in eng.values]
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
        for dst, src in zip(eng.keys, keys):
            dst.copy_(src)
        for dst, src in zip(eng.values, values):
            dst.copy_(src)

    @torch.no_grad()
    def step(self, patch):
        self.patch.copy_(torch.as_tensor(patch, device=self.patch.device).view_as(self.patch))
        self.graph.replay()
        return self.output


class CudaGraphFullTimerXLStep:
    def __init__(self, engine: RollingTimerXLEngine, warmup=3):
        self.engine = engine
        self.warmup = warmup
        self.window = engine.raw_buffer.clone()
        self.graph = self.output = None

    def _body(self):
        eng = self.engine
        embedded, mean, std = eng._embed_full(self.window)
        hidden, keys, values = eng._full_hidden(embedded, store_cache=True)
        eng.raw_buffer.copy_(self.window)
        eng.norm_mean.copy_(mean)
        eng.norm_std.copy_(std)
        for dst, src in zip(eng.keys, keys):
            dst.copy_(src)
        for dst, src in zip(eng.values, values):
            dst.copy_(src)
        return eng._decode(hidden[:, -1], mean, std)

    @torch.no_grad()
    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._body()

    @torch.no_grad()
    def step(self, window):
        self.window.copy_(torch.as_tensor(window, device=self.window.device).view_as(self.window))
        self.graph.replay()
        return self.output
