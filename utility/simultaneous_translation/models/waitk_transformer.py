#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/elbayadm/attn2d/blob/master/examples/waitk
# Implementation of the papers:
#   *Efficient Wait-k Models for Simultaneous Machine Translation
#       http://www.interspeech2020.org/uploadfile/pdf/Tue-1-1-2.pdf

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import checkpoint_utils

from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    base_architecture
)
# user
from simultaneous_translation.modules import (
    WaitkTransformerDecoderLayer,
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("waitk_transformer")
class WaitkTransformerModel(TransformerModel):
    """
    Transformer with a uni-directional encoder and wait-k decoder
    """
    @property
    def pre_decision_ratio(self):
        return self.decoder.pre_decision_ratio  # for speech

    @property
    def waitk(self):
        return self.decoder.waitk

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(WaitkTransformerModel, WaitkTransformerModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument('--waitk', type=int, required=True,
                            help='wait-k for incremental reading')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = CausalTransformerEncoder(
            args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = WaitkTransformerDecoder(
            args, tgt_dict, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
        return decoder

    # def forward_embeddings(self, tokens):
    #     """ convenient function for sinkhorn loss """
    #     return F.embedding(
    #         tokens,
    #         self.decoder.output_projection.weight
    #     )

    def output_projection(self, x):
        """ convenient function"""
        return self.decoder.output_projection(x)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """ convenient override"""
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        x, extra = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
        )
        extra["decoder_states"] = x
        logits = self.decoder.output_projection(x)
        return logits, extra


class CausalTransformerEncoder(TransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [CausalTransformerEncoderLayer(args)
                for i in range(args.encoder_layers)]
        )

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,  # not used
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        incremental_step: Optional[int] = 1,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,  # not used
    ):
        """ Same as parent but with incremental_states """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                src_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            src_tokens = src_tokens[:, -incremental_step:]
            if positions is not None:
                positions = positions[:, -incremental_step:]

        # embed tokens and positions
        x = encoder_embedding = self.embed_scale * \
            self.embed_tokens(src_tokens)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                incremental_state=incremental_state,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def slice_encoder_out(self, encoder_out, context_size):
        """ Slice encoder output according to *context_size*.
        encoder_out:
            (S, N, E) -> (context_size, N, E)
        encoder_padding_mask:
            (N, S) -> (N, context_size)
        encoder_embedding:
            (N, S, E) -> (N, context_size, E)
        encoder_states:
            List(S, N, E) -> List(context_size, N, E)
        """
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x[:context_size].clone() for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x[:, :context_size].clone() for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x[:, :context_size].clone() for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state[:context_size].clone()

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class WaitkTransformerDecoder(TransformerDecoder):
    """
    1. Adds wait-k encoder_masks in training.
    """
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        **kwargs,
    ):
        super().__init__(args, dictionary, embed_tokens, **kwargs)

        self.waitk = args.waitk
        self.pre_decision_ratio = 1  # 1 for text model, > 1 for speech.

    def build_decoder_layer(self, args, no_encoder_attn=False):
        # change to waitk layer.
        return WaitkTransformerDecoderLayer(args, no_encoder_attn)

    def get_attention_mask(self, x, src_len, waitk=None, pre_decision_ratio=None):
        if waitk is None:
            waitk = self.waitk

        if pre_decision_ratio is None:
            pre_decision_ratio = self.pre_decision_ratio

        pooled_src_len = src_len // pre_decision_ratio + 1

        if waitk >= pooled_src_len:
            return None

        neg_inf = -torch.finfo(x.dtype).max
        encoder_attn_mask = torch.triu(
            x.new_full(
                (x.size(0), pooled_src_len),
                neg_inf
            ), waitk
        )
        if waitk <= 0:
            encoder_attn_mask[:, 0] = 0

        # upsample
        encoder_attn_mask = encoder_attn_mask.repeat_interleave(
            pre_decision_ratio, dim=1)[:, :src_len]

        return encoder_attn_mask

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        add encoder_attn_mask (wait-k masking) at training time. otherwise is the same as original.
        """

        bs, slen = prev_output_tokens.size()
        encoder_states: Optional[Tensor] = None
        encoder_padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            encoder_states = encoder_out["encoder_out"][0]
            assert (
                encoder_states.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {encoder_states.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            # full_length = prev_output_tokens.size(1)
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                # training time
                self_attn_mask = self.buffered_future_mask(x)
                encoder_attn_mask = self.get_attention_mask(x, encoder_states.size(0))
            else:
                # inference time
                self_attn_mask = None
                encoder_attn_mask = None
                # encoder_attn_mask = self.get_attention_mask(
                #     x.expand(full_length, -1, -1), encoder_states.size(0))
                # encoder_attn_mask = encoder_attn_mask[-1:, :] if encoder_attn_mask is not None else None

            x, attn = layer(
                x,
                encoder_states,
                encoder_padding_mask,
                encoder_attn_mask=encoder_attn_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clear cache in the monotonic layers.
        The cache is generated because of a forward pass of decode but no prediction.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)


@register_model_architecture(
    "waitk_transformer", "waitk_transformer"
)
def waitk_transformer(args):
    args.waitk = getattr(args, 'waitk', 60000)  # default is wait-until-end

    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)

    args.share_decoder_input_output_embed = True
    base_architecture(args)


@register_model_architecture(
    "waitk_transformer", "waitk_transformer_small"
)
def waitk_transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", True)

    waitk_transformer(args)
