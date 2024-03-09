from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    base_architecture,
)
from fairseq.models.transformer.transformer_base import Embedding
from fairseq.modules import FairseqDropout
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from utils import get_logger
from .egec_language_pair_dataset import ERROR_TYPE_TOKENS, TAGGING_LABELS

LOGGER = get_logger(__name__)


class ExplainableGECTransformerEncoder(TransformerEncoder):
    """ Revised by yejh on 2023.08.10
        1) Introduce src_dropout
        2) Map pointing index
    """

    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        super().__init__(args, dictionary, embed_tokens, return_fc)
        self.src_dropout = args.source_word_dropout
        LOGGER.info(f"Build ExplainableGECTransformerEncoder")
        LOGGER.info(f"Use src_dropout: {self.src_dropout}")

    @classmethod
    def source_dropout(cls, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens * (1 / keep_prob)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        if self.training:
            token_embedding = self.source_dropout(token_embedding, self.src_dropout)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        if self.cfg.explanation_setting == "infusion":
            dict_size = len(self.dictionary)
            pointing_mask = src_tokens.ge(dict_size)  # 需要映射的位置为 True
            pointing_index = src_tokens - dict_size

            if self.left_pad_source:
                pad_idx = self.dictionary.pad()
                pad_len = src_tokens.eq(pad_idx).sum(-1, keepdim=True)  # [B, 1]
                pointing_index = pointing_index + pad_len

            pointing_index = pointing_index.masked_fill(pointing_index.lt(0), 0)  # [B, T]
            # [B, TS], [B, T] -> [B, T]
            pointing_tokens = src_tokens.gather(index=pointing_index, dim=1)
            src_tokens = torch.where(pointing_mask, pointing_tokens, src_tokens)

        return super().forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
        )

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        return super().reorder_encoder_out(encoder_out, new_order)

    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return super().reorder_encoder_out(encoder_out, new_order)


class ExplainableGECTransformerDecoder(TransformerDecoder):
    """ Revised by yejh on 2023.08.10
        1) Incorporating Pointer Network
        2) Extend dictionary to support error type classification
    """

    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            sequence_tagging=False,
            use_encoder_mlp=False,
            use_decoder_mlp=False,
    ):
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        LOGGER.info(f"Build ExplainableGECTransformerDecoder")

        self.sequence_tagging = sequence_tagging
        if sequence_tagging:
            self.num_labels = len(TAGGING_LABELS)
            self.id2label = {i: label for i, label in enumerate(TAGGING_LABELS)}
            self.label2id = {label: i for i, label in enumerate(TAGGING_LABELS)}
            self.tagging_head = TaggingHead(
                input_dim=cfg.encoder_embed_dim,
                inner_dim=cfg.encoder_embed_dim,
                num_classes=self.num_labels,
                activation_fn=cfg.activation_fn,
                dropout=cfg.dropout,
                q_noise=cfg.quant_noise_pq,
                qn_block_size=cfg.quant_noise_pq_block_size,
                do_spectral_norm=False,
            )
            # self.tagging_head = TaggingHead(
            #     input_dim=cfg.encoder_embed_dim,
            #     num_classes=self.num_labels,
            #     dropout=cfg.dropout,
            # )

        self.use_encoder_mlp = use_encoder_mlp
        self.use_decoder_mlp = use_decoder_mlp
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(self.output_embed_dim, self.output_embed_dim, bias=False),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.output_embed_dim, self.output_embed_dim, bias=False),
            )
        if use_decoder_mlp:
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.output_embed_dim, self.output_embed_dim, bias=False),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.output_embed_dim, self.output_embed_dim, bias=False),
            )

        self.dropout_pointer = FairseqDropout(
            0.2,
            module_name=self.__class__.__name__,
        )

    def forward_tagging(self, encoder_out):
        enc = encoder_out["encoder_out"][0]  # [LS, B, D]
        enc = enc.transpose(0, 1)  # [B, LS, D]
        tag_logits = self.tagging_head(enc)  # [B, LS, C]
        # tag_preds = torch.argmax(tag_logits, dim=-1)  # [B, LS]
        # tag_preds = [x.detach().cpu().tolist() for x in tag_preds]
        # for tag in tag_preds:
        #     print(tag)
        return tag_logits

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        # Sequence Tagging (Run only during training and evaluating, rather than inference)
        tag_logits = None
        if self.sequence_tagging and incremental_state is None:
            tag_logits = self.forward_tagging(encoder_out)

        # Map pointing index to token index (Only active for inference)
        if self.cfg.explanation_setting in ["explanation", "rationalization"]:
            dict_size = len(self.dictionary)
            src_tokens = encoder_out["src_tokens"][0]  # [B, LS]
            pointing_mask = prev_output_tokens.ge(dict_size)  # 需要映射的位置为 1
            pointing_index = prev_output_tokens - dict_size

            if self.left_pad_source:
                raise NotImplementedError("Not implemented for inference")
                # pad_idx = self.dictionary.pad()
                # pad_len = src_tokens.eq(pad_idx).sum(-1, keepdim=True)  # [B, 1]
                # pointing_index = pointing_index + pad_len

            pointing_index = pointing_index.masked_fill(pointing_index.lt(0), 0)  # [B, T]
            # [B, TS], [B, T] -> [B, T]
            pointing_tokens = src_tokens.gather(index=pointing_index, dim=1)
            prev_output_tokens = torch.where(pointing_mask, pointing_tokens, prev_output_tokens)

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x, encoder_out=encoder_out)
        return x, extra, tag_logits

    def output_layer(
            self,
            features,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            bpe_first: Optional[Tensor] = None,
    ):
        """ Project features to the vocabulary and pointing (src_tokens) size.
            @param features: [B, LT, H]: outputs of Decoder
            @param encoder_out: output of Encoder
            @param bpe_first: [B, LS_word] 显示每个单词第一个 BPE
        """
        # Project features to the vocabulary size
        vocab_logits = self.output_projection(features)  # [B, LT, V]

        if self.cfg.explanation_setting not in ["rationalization", "explanation"]:
            return vocab_logits

        # Project features to the pointing (src_tokens) size
        # src_tokens = encoder_out["src_tokens"]  # [B, LS]
        enc = encoder_out["encoder_out"][0].transpose(1, 0)  # [B, LS, H]
        encoder_embedding = encoder_out["encoder_embedding"][0]  # [B, LS, H]
        encoder_padding_mask = encoder_out["encoder_padding_mask"][0]  # [B, LS]

        if hasattr(self, "encoder_mlp"):
            enc = self.encoder_mlp(enc)

        x = self.dropout_pointer(features)  # [B, LT, H]
        encoder_embedding = self.dropout_pointer(encoder_embedding)  # [B, LS, H]

        if hasattr(self, "decoder_mlp"):
            x = self.decoder_mlp(x)

        enc = (enc + encoder_embedding) / 2
        point_scores = torch.einsum('blh,bnh->bln', x, enc)  # [B, LT, LS]

        point_scores = point_scores.masked_fill(
            encoder_padding_mask.unsqueeze(1),  # [B, 1, LS]
            -1e+3,
        )  # [B, LT, LS]

        logits = torch.concat((vocab_logits, point_scores), dim=-1)  # [B, LT, V+LS]
        return logits

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """ Revised by yejh on 2023.09.10
            Get normalized probabilities (or log probs) from a net's output.
        """
        if len(net_output) == 2:
            return super().get_normalized_probs_scriptable(net_output, log_probs, sample)

        logits, tag_logits = net_output[0], net_output[2]
        if log_probs:
            return (
                utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace),
                utils.log_softmax(tag_logits, dim=-1, onnx_trace=self.onnx_trace)
                if tag_logits is not None else None,
            )
        else:
            return (
                utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace),
                utils.softmax(tag_logits, dim=-1, onnx_trace=self.onnx_trace)
                if tag_logits is not None else None,
            )


@register_model("explainable_gec_transformer")
class ExplainableGECTransformer(TransformerModel):
    """ 相比传统的 Transformer 的改动：
        1) 使用 ExplainableGECTransformerEncoder
        2) 使用 ExplainableGECTransformerDecoder
        3) 提供 set_beam_size 方法支持 gec_dev 实时评估 F0.5
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--source-word-dropout",
            type=float, metavar="D", default=0.2,
            help="dropout probability for source word dropout",
        )
        parser.add_argument(
            "--sequence-tagging",
            action="store_true", default=False,
            help="carry on sequence tagging task",
        )
        parser.add_argument(
            "--use-encoder-mlp",
            action="store_true", default=False,
            help="use mlp to further handle the encoder output",
        )
        parser.add_argument(
            "--use-decoder-mlp",
            action="store_true", default=False,
            help="use mlp to further handle the decoder output",
        )

    @classmethod
    def build_model(cls, args, task):
        LOGGER.info(f"left_pad_source: {task.cfg.left_pad_source}")
        model = super().build_model(args, task)
        model.task = task
        model.encoder.left_pad_source = task.cfg.left_pad_source
        model.decoder.left_pad_source = task.cfg.left_pad_source
        return model

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary) + len(ERROR_TYPE_TOKENS) + 1
        LOGGER.info(f"num_embeddings: {num_embeddings}")
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return ExplainableGECTransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return ExplainableGECTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            use_encoder_mlp=cfg.use_encoder_mlp,
            use_decoder_mlp=cfg.use_decoder_mlp,
            sequence_tagging=cfg.sequence_tagging,
        )

    def set_beam_size(self, beam):
        """Set beam size for efficient beamable enc-dec attention."""
        beamable = False
        for layer in self.decoder.layers:
            if layer.encoder_attn is not None:
                if hasattr(layer.encoder_attn, "set_beam_size"):
                    layer.encoder_attn.set_beam_size(beam)
                    beamable = True
        if beamable:
            self.encoder.reorder_encoder_out = self.encoder._reorder_encoder_out

    def get_targets(self, sample, net_output):
        """ Revised by yejh on 2023.08.18
            Get targets from either the sample or the net's output.
            1) Correction Target
            2) Tagging Target
        """
        return (
            sample["target"],
            sample["tagging_target"] if "tagging_target" in sample else None,
        )

    # def get_targets(self, sample, net_output):
    #     """ Revised by yejh on 2023.08.18
    #         Get targets from either the sample or the net's output.
    #     """
    #     target = sample["target"]
    #
    #     if not self.task.cfg.left_pad_source:
    #         return target
    #     if self.cfg.explanation_setting not in ["explanation", "rationalization"]:
    #         return target
    #
    #     # 因为 src_tokens 默认是 left_pad 的，所以在训练过程需要考虑 pad_len
    #     src_tokens = sample["net_input"]["src_tokens"]  # [B, LS]
    #     pad_idx = self.encoder.dictionary.pad()
    #     pad_len = src_tokens.eq(pad_idx).sum(-1, keepdim=True)  # [B, 1]
    #
    #     pointing_mask = target.ge(len(self.encoder.dictionary))  # 需要映射的位置为 1
    #     target = torch.where(pointing_mask, target + pad_len, target)
    #     return target


# class TaggingHead(nn.Module):
#     """ Head for token-level classification tasks. """
#
#     def __init__(
#             self,
#             input_dim,
#             num_classes,
#             dropout,
#     ):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.out_proj =nn.Linear(input_dim, num_classes)
#         # self.out_proj = apply_quant_noise_(
#         #     nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
#         # )
#
#     def forward(self, x):
#         # [LS, B, D]
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

class TaggingHead(nn.Module):
    """ Head for token-level classification tasks. """

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            dropout,
            q_noise=0,
            qn_block_size=8,
            do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        # self.out_proj = apply_quant_noise_(
        #     nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        # )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, x):
        # [LS, B, D]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("explainable_gec_transformer", "explainable_gec_transformer")
def gec_transformer_base_architecture(args):
    base_architecture(args)
    args.source_word_dropout = getattr(args, "source_word_dropout", 0.0)
    args.use_encoder_mlp = getattr(args, "use_encoder_mlp", False)
