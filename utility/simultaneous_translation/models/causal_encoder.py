#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoderModel,
)

# user
from simultaneous_translation.models.sinkhorn_encoder import (
    SinkhornEncoderModel,
    sinkhorn_encoder
)

logger = logging.getLogger(__name__)


@register_model("causal_encoder")
class CausalEncoderModel(SinkhornEncoderModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        FairseqEncoderModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
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

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **unused):

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


@register_model_architecture(
    "causal_encoder", "causal_encoder"
)
def causal_encoder(args):
    args.non_causal_layers = 0
    args.sinkhorn_tau = 1
    args.sinkhorn_iters = 1
    args.sinkhorn_noise_factor = 0
    args.sinkhorn_bucket_size = 1
    args.sinkhorn_energy = "dot"
    args.mask_ratio = 1
    args.mask_uniform = False

    sinkhorn_encoder(args)


@register_model_architecture(
    "causal_encoder", "causal_encoder_small"
)
def causal_encoder_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    causal_encoder(args)
