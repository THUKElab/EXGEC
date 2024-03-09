import torch
import torch.nn as nn
from typing import Optional

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.bart import (
    BARTHubInterface,
    bart_large_architecture,
    bart_base_architecture,
)
from .egec_transformer import ExplainableGECTransformer
from .egec_language_pair_dataset import (
    SEPERATOR_TOKEN,
    ERROR_TYPE_TOKENS,
)
from utils import get_logger

LOGGER = get_logger(__name__)


@register_model("egec_bart")
class ExplainableGECBARTModel(ExplainableGECTransformer):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()
        self.sep_index = None

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            features_only: bool = False,
            classification_head_name: Optional[str] = None,
            token_embeddings: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
        )
        encoder_out["src_tokens"] = [src_tokens]
        logits, extra, tag_logits = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra, tag_logits

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="gpt2",
            sample_break_mode="eos",
            **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    # def load_bart_state_dict_from_transformers(self, model, strict=True, args=None):
    #     """ Copies parameters and buffers from *state_dict* into this module and its descendants.
    #         Overrides the method in :class:`nn.Module`. Compared with that method
    #         this additionally "upgrades" *state_dicts* from old checkpoints.
    #     """
    #     new_state_dict = {}
    #     for k, v in model.named_parameters():
    #         new_state_dict[k.replace("model.", "")] = v
    #     # Share all embeddings
    #     shared_weight = self._get_resized_embeddings(new_state_dict["shared.weight"])
    #     print(f"shared_weight: {new_state_dict['shared.weight'].size()}")
    #
    #     # Default: share all embeddings
    #     new_state_dict["encoder.embed_tokens.weight"] = new_state_dict["decoder.embed_tokens.weight"] \
    #         = new_state_dict["decoder.output_projection.weight"] = shared_weight
    #     del new_state_dict["shared.weight"]
    #     del model
    #     return super().load_state_dict(new_state_dict, True)
    #
    # def _get_resized_embeddings(
    #         self, old_embeddings: torch.nn.Embedding, new_num_tokens: Optional[int] = None
    # ) -> torch.nn.Embedding:
    #     """
    #     Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
    #     initialized vectors at the end. Reducing the size will remove vectors from the end
    #
    #     Args:
    #         old_embeddings (:obj:`torch.nn.Embedding`):
    #             Old embeddings to be resized.
    #         new_num_tokens (:obj:`int`, `optional`):
    #             New number of tokens in the embedding matrix.
    #
    #             Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
    #             vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
    #             :obj:`torch.nn.Embedding`` module of the model without doing anything.
    #
    #     Return:
    #         :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
    #         :obj:`new_num_tokens` is :obj:`None`
    #     """
    #     old_num_tokens, old_embedding_dim = old_embeddings.size()
    #     if old_num_tokens == new_num_tokens:
    #         return old_embeddings
    #
    #     padding_idx = 1
    #     new_num_tokens = old_num_tokens + 4
    #     # Build new embeddings
    #     new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    #
    #     # initialize all new embeddings (in particular added tokens)
    #     nn.init.normal_(new_embeddings.weight, mean=0, std=old_embedding_dim ** -0.5)
    #     nn.init.constant_(new_embeddings.weight[padding_idx], 0)
    #
    #     # Copy token embeddings from the previous weights
    #     num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    #     new_embeddings.weight.data[-num_tokens_to_copy:, :] = old_embeddings.data[:num_tokens_to_copy, :]
    #
    #     return new_embeddings.weight

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        # Init sequence tagging head
        if self.decoder.sequence_tagging:
            dim = self.decoder.output_embed_dim
            out_proj = state_dict["decoder.output_projection.weight"]
            state_dict["decoder.tagging_head.dense.weight"] = out_proj.new(dim, dim)
            state_dict["decoder.tagging_head.dense.bias"] = out_proj.new(dim, ).fill_(0.0)
            # [num_label, D]
            state_dict["decoder.tagging_head.out_proj.weight"] = out_proj.new(self.decoder.num_labels, dim)
            state_dict["decoder.tagging_head.out_proj.bias"] = out_proj.new(self.decoder.num_labels, ).fill_(0.0)
            # Init parameters as TransformerDecoderBase.build_output_projection
            nn.init.normal_(state_dict["decoder.tagging_head.dense.weight"], mean=0, std=dim ** -0.5)
            nn.init.normal_(state_dict["decoder.tagging_head.out_proj.weight"], mean=0, std=dim ** -0.5)

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]
                LOGGER.info(f"Truncate Embedding {key}: {state_dict[key].size()}")

        def add_token(key, new_token, init_token=None):
            if key in state_dict:
                if init_token is None:
                    new_embedding = state_dict[key].new(1, state_dict[key].size(1))
                    nn.init.normal_(new_embedding, mean=0, std=state_dict[key].size(1) ** -0.5)
                else:
                    new_embedding = state_dict[key][init_token, :].clone().detach().unsqueeze(0)
                state_dict[key] = torch.cat((state_dict[key], new_embedding), dim=0)
                LOGGER.info(f"Add Embedding {new_token} {key}: {state_dict[key].size()}")

        # for i in range(len(self.encoder.dictionary) - 10, len(self.encoder.dictionary)):
        #     print(self.encoder.dictionary[i])

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
                loaded_dict_size == len(self.encoder.dictionary) + 1
                and "<mask>" not in self.encoder.dictionary
        ):
            loaded_dict_size -= 1
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        if loaded_dict_size != len(self.encoder.dictionary):  # For inference
            assert loaded_dict_size == len(self.encoder.dictionary) + len(ERROR_TYPE_TOKENS) + 1
            sep_index1 = self.task.src_dict.add_symbol(SEPERATOR_TOKEN)
            sep_index2 = self.task.tgt_dict.add_symbol(SEPERATOR_TOKEN)
            assert sep_index1 == sep_index2

            for k, v in ERROR_TYPE_TOKENS.items():
                if self.task.src_dict.index(v) != self.task.src_dict.unk():
                    LOGGER.info(f"Add Error Type {v}: {len(self.src_dict)}")

                error_types_index1 = self.task.src_dict.add_symbol(v)
                error_types_index2 = self.task.tgt_dict.add_symbol(v)
                assert error_types_index1 == error_types_index2, \
                    f"Invalid dictionary: {v} | {error_types_index1} | {error_types_index2}"
            return

        # Add seperator token <sep>
        self.sep_index = self.encoder.dictionary.add_symbol(SEPERATOR_TOKEN)
        sep_index = self.decoder.dictionary.add_symbol(SEPERATOR_TOKEN)
        assert self.sep_index == sep_index

        unk_index = self.encoder.dictionary.unk_index
        add_token("encoder.embed_tokens.weight", SEPERATOR_TOKEN, unk_index)
        add_token("decoder.embed_tokens.weight", SEPERATOR_TOKEN, unk_index)
        add_token("encoder.output_projection.weight", SEPERATOR_TOKEN, unk_index)
        add_token("decoder.output_projection.weight", SEPERATOR_TOKEN, unk_index)

        # Add error_type token like <Infinitives>
        valid_dataset = self.task.datasets["valid"]
        valid_dataset.init_error_types()
        for k, v in valid_dataset.error_types_index.items():
            add_token("encoder.embed_tokens.weight", k, unk_index)
            add_token("decoder.embed_tokens.weight", k, unk_index)
            add_token("encoder.output_projection.weight", k, unk_index)
            add_token("decoder.output_projection.weight", k, unk_index)

        # Init encoder_mlp
        if self.decoder.use_encoder_mlp:
            dim = self.decoder.output_embed_dim
            state_dict["decoder.encoder_mlp.0.weight"] = state_dict["decoder.output_projection.weight"].new(dim, dim)
            state_dict["decoder.encoder_mlp.3.weight"] = state_dict["decoder.output_projection.weight"].new(dim, dim)
            nn.init.normal_(state_dict["decoder.encoder_mlp.0.weight"], mean=0, std=dim ** -0.5)
            nn.init.normal_(state_dict["decoder.encoder_mlp.3.weight"], mean=0, std=dim ** -0.5)
        if self.decoder.use_decoder_mlp:
            dim = self.decoder.output_embed_dim
            state_dict["decoder.decoder_mlp.0.weight"] = state_dict["decoder.output_projection.weight"].new(dim, dim)
            state_dict["decoder.decoder_mlp.3.weight"] = state_dict["decoder.output_projection.weight"].new(dim, dim)
            nn.init.normal_(state_dict["decoder.decoder_mlp.0.weight"], mean=0, std=dim ** -0.5)
            nn.init.normal_(state_dict["decoder.decoder_mlp.3.weight"], mean=0, std=dim ** -0.5)


@register_model_architecture("egec_bart", "egec_bart_large")
def egec_bart_large_architecture(args):
    bart_large_architecture(args)
    # To alleviate over-fitting (2018 NAACL)
    args.source_word_dropout = getattr(args, "source_word_dropout", 0.0)
    args.use_encoder_mlp = getattr(args, "use_encoder_mlp", False)


@register_model_architecture("egec_bart", "egec_bart_base")
def egec_bart_base_architecture(args):
    bart_base_architecture(args)
    egec_bart_large_architecture(args)
