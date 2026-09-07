import warnings
from typing import Any, Dict, List, Optional, Union

import torch

from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation import validate_stopping_criteria, EosTokenCriteria
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput


class TSGenerationMixin(GenerationMixin):

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        input_ids_origin_device = input_ids.device
        input_ids = input_ids.to(self.device)
        if len(input_ids.shape) == 2:
            batch_size, cur_len = input_ids.shape
        else:
            raise ValueError('Input shape must be: [batch_size, seq_len]')
        
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        
        eos_token_id = generation_config._eos_token_tensor
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        max_length = stopping_criteria.max_length
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            input_length = input_ids.shape[1]

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                max_horizon_length=max_length - input_length,
            )

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            next_tokens = next_tokens_scores

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            next_tokens = next_tokens.reshape(batch_size, -1, self.config.input_size)
            horizon_length = next_tokens.shape[1]

            if input_ids.ndim == 2:
                input_ids = input_ids.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-2)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids[..., 0], scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        if streamer is not None:
            streamer.end()

        input_ids.squeeze_(dim=-1).to(input_ids_origin_device)
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            horizon_length: int = 1,
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = outputs.past_key_values
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], horizon_length))], dim=-1
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], horizon_length))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + horizon_length

        return model_kwargs
