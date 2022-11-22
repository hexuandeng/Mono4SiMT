#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils, utils

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
    FairseqEncoderModel,
)
from fairseq.models.transformer import (
    TransformerEncoder,
    Embedding,
    Linear,
    base_architecture,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

# user
from simultaneous_translation.models.nat_utils import (
    generate,
    inject_noise
)
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
    SinkhornAttention,
)

logger = logging.getLogger(__name__)


@register_model("sinkhorn_encoder")
class SinkhornEncoderModel(FairseqEncoderModel):
    """
    causal encoder + ASN + output projection
    """

    def __init__(self, encoder, output_projection):
        super().__init__(encoder)
        self.output_projection = output_projection
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SinkhornEncoderModel, SinkhornEncoderModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--non-causal-layers",
            type=int,
            help=(
                'number of layers for non-causal encoder.'
            ),
        )
        parser.add_argument(
            '--sinkhorn-tau',
            type=float,
            required=True,
            help='temperature for gumbel sinkhorn.'
        )
        parser.add_argument(
            "--sinkhorn-iters",
            type=int,
            required=True,
            help=(
                'iters of sinkhorn normalization to perform.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-noise-factor",
            type=float,
            required=True,
            help=(
                'represents how many gumbel randomness in training.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-bucket-size",
            type=int,
            required=True,
            help=(
                'number of elements to group before performing sinkhorn sorting.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-energy",
            type=str,
            required=True,
            choices=["dot", "cos", "l2"],
            help=(
                'type of energy function to use to calculate attention. available: dot, cos, L2'
            ),
        )
        parser.add_argument(
            "--upsample-ratio",
            type=int,
            help=(
                'number of upsampling factor before ctc loss. used for mt.'
            ),
        )
        parser.add_argument(
            '--delay', type=int, help='delay for incremental reading')

        parser.add_argument(
            "--mask-ratio",
            required=True,
            type=float,
            help=(
                'ratio of target tokens to mask when feeding to sorting network.'
            ),
        )
        parser.add_argument(
            "--mask-uniform",
            action="store_true",
            default=False,
            help=(
                'ratio of target tokens to mask when feeding to aligner.'
            ),
        )

    @classmethod
    def build_encoder(cls, args, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens):
        encoder = CausalTransformerEncoder(args, src_dict, encoder_embed_tokens)
        encoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        cascade = ASNAugmentedEncoder(args, encoder, tgt_dict, decoder_embed_tokens)
        return cascade

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        encoder_embed_tokens = cls.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        output_projection = nn.Linear(
            decoder_embed_tokens.weight.shape[1],
            decoder_embed_tokens.weight.shape[0],
            bias=False
        )
        output_projection.weight = decoder_embed_tokens.weight

        encoder = cls.build_encoder(
            args, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        return cls(encoder, output_projection)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        logits = net_output[0]

        if torch.is_tensor(logits):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits_f = logits.float()
            if log_probs:
                lprobs = F.log_softmax(logits_f, dim=-1)
            else:
                lprobs = F.softmax(logits_f, dim=-1)
        else:
            raise NotImplementedError

        return lprobs

    def forward_causal(
        self, src_tokens, src_lengths,
        return_all_hiddens: bool = False,
        **unused,
    ):
        encoder_out = self.encoder.forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        x = self.output_projection(encoder_out["encoder_out"][0])
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out,
            "attn": encoder_out["attn"],
            "log_alpha": encoder_out["log_alpha"],
        }
        return x, extra

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: bool = False, **unused):

        encoder_out = self.encoder.forward_train(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens,
            return_all_hiddens=return_all_hiddens
        )
        x = self.output_projection(encoder_out["encoder_out"][0])
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out,
            "attn": encoder_out["attn"],
            "log_alpha": encoder_out["log_alpha"],
        }
        return x, extra

    @property
    def output_layer(self):
        return self.output_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        # if not from_encoder:
        #     return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx)

    def max_decoder_positions(self):
        """Used by sequence generator."""
        return self.encoder.max_positions()


class CausalTransformerEncoder(TransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.delay = args.delay
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(args, delay=args.delay)
        ])
        self.layers.extend(
            [CausalTransformerEncoderLayer(args)
             for i in range(args.encoder_layers - 1)]
        )

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,  # not used
        incremental_state: Optional[Dict[str,
                                         Dict[str, Optional[Tensor]]]] = None,
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
            # incremental_state for embed_positions is designed for single step.
            # # slow
            # positions = self.embed_positions(
            #     src_tokens,  # incremental_state=incremental_state
            # )
            # fast
            positions = self.embed_positions(
                src_tokens,
                incremental_state=incremental_state,
                timestep=torch.LongTensor(
                    [src_tokens.size(1) - incremental_step])
            )
            if incremental_step > 1:
                for i in range(1, incremental_step):
                    timestep = src_tokens.size(1) - incremental_step + i
                    positions = torch.cat(
                        (
                            positions,
                            self.embed_positions(
                                src_tokens,
                                incremental_state=incremental_state,
                                timestep=torch.LongTensor([timestep])
                            )
                        ), dim=1
                    )

        if incremental_state is not None:
            src_tokens = src_tokens[:, -incremental_step:]
            if positions is not None:
                positions = positions[:, -incremental_step:]
            has_pads = False

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

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
        keep: Optional[int] = None,
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
                layer.prune_incremental_state(incremental_state, keep)

    def load_state_dict(self, state_dict, strict=True):
        """
        1. remove ``causal_encoder'' from the state_dict keys.
        2. ignores upsampler and decoder_embed.
        """
        changes = re.compile("causal_encoder.")
        ignores = ["upsampler", "decoder_embed"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if any([i in k for i in ignores]):
                continue
            new_state_dict[changes.sub("", k)] = v

        return super().load_state_dict(new_state_dict, strict=strict)


class ASNAugmentedEncoder(FairseqEncoder):
    """
    Add following layers to the causal encoder,
    1) several non-causal encoder layers
    2) 1 sinkhorn attention
    """

    def __init__(self, args, causal_encoder, tgt_dict, decoder_embed_tokens):
        super().__init__(None)
        self.causal_encoder = causal_encoder
        self.non_causal_layers = nn.ModuleList([
            TransformerDecoderLayer(args) for i in range(args.non_causal_layers)
        ])
        export = getattr(args, "export", False)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim, export=export)
        else:
            self.layer_norm = None
        self.sinkhorn_layer = SinkhornAttention(
            args.encoder_embed_dim,
            bucket_size=args.sinkhorn_bucket_size,
            dropout=0,  # args.attention_dropout, already have gumbel noise
            no_query_proj=True,
            no_key_proj=True,
            no_value_proj=True,
            no_out_proj=True,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_noise_factor=args.sinkhorn_noise_factor,
            energy_fn=args.sinkhorn_energy,
        )
        self.upsample_ratio = args.upsample_ratio
        if self.upsample_ratio > 1:
            self.upsampler = Linear(
                args.encoder_embed_dim, args.encoder_embed_dim * self.upsample_ratio)

        # below are for input target feeding
        self.mask_ratio = args.mask_ratio
        self.mask_uniform = args.mask_uniform

        self.tgt_dict = tgt_dict
        self.decoder_embed_tokens = decoder_embed_tokens

        self.decoder_embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            args.decoder_embed_dim)
        self.decoder_embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,  # this pos is for target tokens
                args.encoder_embed_dim,
                tgt_dict.pad(),
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def upsample(self, x, encoder_padding_mask):
        if self.upsample_ratio == 1:
            return x, encoder_padding_mask

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                self.upsample_ratio, dim=1)

        T, B, C = x.size()
        # T x B x C
        # -> T x B x C*U
        # -> U * (T x B x C)
        # -> T x U x B x C
        # -> T*U x B x C
        x = torch.stack(
            torch.chunk(
                self.upsampler(x),
                self.upsample_ratio,
                dim=-1
            ),
            dim=1
        ).view(-1, B, C)
        return x, encoder_padding_mask

    def forward(
        self, src_tokens, src_lengths,
        return_all_hiddens: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        incremental_step: Optional[int] = 1,
    ):
        causal_out = self.causal_encoder(
            src_tokens, src_lengths,
            return_all_hiddens=return_all_hiddens,
            incremental_state=incremental_state,
            incremental_step=incremental_step
        )
        x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_out.update({
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
            "attn": [],
            "log_alpha": [],
        })
        return causal_out

    def forward_train(
        self, src_tokens, src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = False
    ):
        """ Added forwards for non-causal and sinkhorn attention """
        causal_out = self.causal_encoder(src_tokens, src_lengths, return_all_hiddens=return_all_hiddens)

        # causal outputs
        causal_states = x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        encoder_states = causal_out["encoder_states"]

        # target feeding (noise + pos emb)
        prev_tokens, prev_padding_mask = inject_noise(
            prev_output_tokens,
            self.tgt_dict,
            ratio=self.mask_ratio if self.training else 0,
            uniform=self.mask_uniform,
        )

        prev_states = self.decoder_embed_scale * self.decoder_embed_tokens(prev_tokens)
        prev_states += self.decoder_embed_positions(prev_tokens)
        prev_states = prev_states.transpose(0, 1)

        # forward non-causal layers
        for layer in self.non_causal_layers:
            # x = layer(x, encoder_padding_mask)
            x, _, _ = layer(
                x,
                prev_states,
                prev_padding_mask,
                self_attn_padding_mask=encoder_padding_mask,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # reorder using sinkhorn layers
        # (q,k,v) = (non-causal, causal, causal)
        x, attn, log_alpha = self.sinkhorn_layer(
            x,  # this is non_causal_states
            causal_states,
            causal_states,
            encoder_padding_mask,
        )

        # upsample
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_states, _ = self.upsample(causal_states, None)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "attn": [attn],
            "log_alpha": [log_alpha],
            "causal_out": [causal_states],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """ This reorder is fairseq's reordering of batch dimension,
        different from our reorder.
        """
        return self.causal_encoder.reorder_encoder_out(encoder_out, new_order)


@register_model_architecture(
    "sinkhorn_encoder", "sinkhorn_encoder"
)
def sinkhorn_encoder(args):
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)
    args.non_causal_layers = getattr(args, "non_causal_layers", 3)

    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.dropout = getattr(args, "dropout", 0.1)

    args.share_decoder_input_output_embed = True

    args.upsample_ratio = getattr(args, "upsample_ratio", 2)
    args.delay = getattr(args, "delay", 1)

    base_architecture(args)


@register_model_architecture(
    "sinkhorn_encoder", "sinkhorn_encoder_small"
)
def sinkhorn_encoder_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.decoder_embed_dim = args.encoder_embed_dim
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)
    args.non_causal_layers = getattr(args, "non_causal_layers", 3)

    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.dropout = getattr(args, "dropout", 0.1)

    args.share_decoder_input_output_embed = True

    args.upsample_ratio = getattr(args, "upsample_ratio", 2)
    args.delay = getattr(args, "delay", 1)

    base_architecture(args)
