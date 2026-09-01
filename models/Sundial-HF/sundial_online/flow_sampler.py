"""Deterministic, CUDA-Graph-safe Sundial TimeFlow sampling."""

from __future__ import annotations

import torch


class FixedNoiseFlowSampler:
    """Run the official Euler sampler with a reusable common noise bank."""

    def __init__(
        self,
        flow_loss,
        batch_size: int,
        num_samples: int,
        num_steps: int,
        device: str,
        seed: int,
        noise_mode: str = "antithetic",
    ):
        if not 1 <= num_steps <= int(flow_loss.num_sampling_steps):
            raise ValueError(
                f"num_steps must be in [1, {flow_loss.num_sampling_steps}]"
            )
        self.flow_loss = flow_loss
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_steps = num_steps
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        if noise_mode == "random":
            noise = torch.randn(
                num_samples,
                batch_size,
                flow_loss.in_channels,
                device=device,
                generator=generator,
            )
        elif noise_mode == "antithetic":
            pair_count = num_samples // 2
            base = torch.randn(
                pair_count,
                batch_size,
                flow_loss.in_channels,
                device=device,
                generator=generator,
            )
            parts = []
            if num_samples % 2:
                parts.append(
                    torch.zeros(
                        1, batch_size, flow_loss.in_channels, device=device
                    )
                )
            for index in range(pair_count):
                parts.extend((base[index : index + 1], -base[index : index + 1]))
            noise = torch.cat(parts, dim=0)
        else:
            raise ValueError("noise_mode must be 'antithetic' or 'random'")
        self.noise = noise.reshape(
            batch_size * num_samples, flow_loss.in_channels
        )
        timesteps = [
            torch.full(
                (batch_size * num_samples,),
                i * 1000.0 / num_steps,
                device=device,
            )
            for i in range(num_steps)
        ]
        with torch.no_grad():
            self.time_embeddings = [
                flow_loss.net.time_embed(timestep).detach() for timestep in timesteps
            ]

    def _velocity(self, x, time_embedding, condition_embedding):
        """Equivalent to flow_loss.net(), with loop invariants hoisted."""
        net = self.flow_loss.net
        hidden = net.input_proj(x)
        modulation = time_embedding + condition_embedding
        for block in net.res_blocks:
            hidden = block(hidden, modulation)
        return net.final_layer(hidden, modulation)

    def sample(self, hidden: torch.Tensor) -> torch.Tensor:
        condition = hidden.repeat(self.num_samples, 1)
        condition_embedding = self.flow_loss.net.cond_embed(condition)
        x = self.noise + 0.0
        dt = 1.0 / self.num_steps
        for time_embedding in self.time_embeddings:
            prediction = self._velocity(x, time_embedding, condition_embedding)
            x = x + (prediction - self.noise) * dt
        return x.reshape(
            self.num_samples, self.batch_size, self.flow_loss.in_channels
        ).transpose(0, 1)

    def point_forecast(self, hidden: torch.Tensor, horizon: int) -> torch.Tensor:
        samples = self.sample(hidden)[:, :, :horizon]
        return samples.median(dim=1).values
