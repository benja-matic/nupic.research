# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import logging
import re

from torch import nn
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TOKENIZER_MAPPING,
    BertConfig,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertTokenizer,
    BertTokenizerFast,
)
from transformers.modeling_utils import (
    PoolerAnswerClass,
    PoolerEndLogits,
    PoolerStartLogits,
)

from transformers.models.bert.modeling_bert import BertOnlyMLMHead

# This is the dict of the `models` module. Anything imported from `models`
# must be contained in this dict. Hence, it will be updated to include
# the new modules made below so they may be imported.
from . import __dict__ as __models_dict__

# Keep a copy of the mappings so they may be directly accessed.
__models_dict__["CONFIG_MAPPING"] = CONFIG_MAPPING
__models_dict__["MODEL_FOR_MASKED_LM_MAPPING"] = MODEL_FOR_MASKED_LM_MAPPING
__models_dict__["MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING"] = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING  # noqa E501
__models_dict__["MODEL_FOR_QUESTION_ANSWERING"] = MODEL_FOR_QUESTION_ANSWERING_MAPPING  # noqa E501
__models_dict__["TOKENIZER_MAPPING"] = TOKENIZER_MAPPING


def register_bert_model(bert_cls):
    """
    This function wraps a BertModel inherited cls and automatically:
        1. Creates an associated BertConfig
        2. Creates an associated BertForMaskedLM
        3. Creates an associated BertForSequenceClassification
        4. Creates an associated BertForQuestionAnswering
        5. Registers these classes with Transformers model mappings

    This last step ensures that the resulting config and models may be used by
    AutoConfig, AutoModelForMaskedLM, and AutoModelForSequenceClassification.

    Assumptions are made to auto-name these classes and the corresponding model type.
    For instance, SparseBertModel will have model_type="sparse_bert" and associated
    classes like SparseBertConfig.

    To customize the the inputs to the model's config, include the dataclass
    `bert_cls.ConfigKWargs`. This is, in fact, required. Upon initialization of the
    config, the fields of that dataclass will be used to extract extra keyword arguments
    and assign them as attributes to the config.

    Example
    ```
    @register_bert_model
    class SparseBertModel(BertModel):

        @dataclass
        class ConfigKWargs:
            # Keyword arguments to configure sparsity.
            sparsity: float = 0.9

        # Define __init__, ect.
        ...

    # Model is ready to auto load.
    config = AutoConfig.for_model("sparse_bert", sparsity=0.5)
    model = AutoModelForMaskedLM.from_config(model)

    config.sparsity
    >>> 0.5

    type(model)
    >>> SparseBertModelForMaskedLM
    """

    assert bert_cls.__name__.endswith("BertModel")

    # Get first part of name e.g. StaticSparseBertModel -> StaticSparse
    name_prefix = bert_cls.__name__.replace("BertModel", "")

    # Create new bert config and models based off of `bert_cls`.
    config_cls = create_config_class(bert_cls, name_prefix)
    masked_lm_cls = create_masked_lm_class(bert_cls, name_prefix)
    seq_classification_cls = create_sequence_classification_class(bert_cls, name_prefix)
    question_answering_cls = create_question_answering_class(bert_cls, name_prefix)

    # Specify the correct config class
    bert_cls.config_class = config_cls
    masked_lm_cls.config_class = config_cls
    seq_classification_cls.config_class = config_cls
    question_answering_cls.config_class = config_cls

    # Update Transformers mappings to auto-load these new models.
    CONFIG_MAPPING.update({
        config_cls.model_type: config_cls
    })
    TOKENIZER_MAPPING.update({
        config_cls: (BertTokenizer, BertTokenizerFast),
    })
    MODEL_FOR_MASKED_LM_MAPPING.update({
        config_cls: masked_lm_cls,
    })
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.update({
        config_cls: seq_classification_cls
    })
    MODEL_FOR_QUESTION_ANSWERING_MAPPING.update({
        config_cls: question_answering_cls
    })

    # Update the `models` modules so that these classes may be imported.
    __models_dict__.update({
        config_cls.__name__: config_cls,
        masked_lm_cls.__name__: masked_lm_cls,
        seq_classification_cls.__name__: seq_classification_cls,
        question_answering_cls.__name__: question_answering_cls,
    })


def create_config_class(bert_cls, name_prefix):
    assert hasattr(bert_cls, "ConfigKWargs")
    config_dataclass = bert_cls.ConfigKWargs
    dataclass_fields = config_dataclass.__dataclass_fields__

    class NewBertConfig(BertConfig):
        """
        This is the configuration class to store the configuration of a
        {name_prefix}BertModel. This is adapted from `BertConfig`_.

        .. _BertConfig:
            https://huggingface.co/transformers/_modules/transformers/models/bert/configuration_bert.html#BertConfig
        """

        def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
        ):
            super().__init__(pad_token_id=pad_token_id, **kwargs)

            # Bert config params.
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.gradient_checkpointing = gradient_checkpointing
            self.position_embedding_type = position_embedding_type
            self.use_cache = use_cache

            # Use the dataclass fields to define extra config parameters.
            extra_config = {}
            for arg in dataclass_fields.keys():
                if arg in kwargs:
                    extra_config[arg] = kwargs[arg]

            extra_config = config_dataclass(**extra_config).__dict__
            self.__dict__.update(extra_config)

    # Format model_type e.g. "StaticSparse" will become "static_sparse_bert"
    name_segments = re.findall("[A-Z][^A-Z]*", name_prefix)
    model_type = "_".join(name_segments).lower()
    model_type += "_bert"

    # Rename new config class, format docstring, and add model_type.
    new_cls = NewBertConfig
    new_cls.__name__ = new_cls.__name__.replace("New", name_prefix)
    new_cls.__doc__ = new_cls.__doc__.format(name_prefix=name_prefix)
    new_cls.model_type = model_type

    return new_cls


def create_masked_lm_class(bert_cls, name_prefix):
    """
    Create a BertForMaskedLM that calls `bert_cls` in it's forward.
    """

    class NewBertForMaskedLM(BertForMaskedLM):
        """
        Sparse Bert Model with a `language modeling` head on top. Calls
        {name_prefix}BertModel in its forward.

        This is adapted from `BertForMaskedLM`_

        .. _SparseBertForMaskedLM:
            https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForMaskedLM
        """

        def __init__(self, config):

            # Call the init one parent class up.
            # Otherwise, the model will be defined twice.
            BertPreTrainedModel.__init__(self, config)

            if config.is_decoder:
                logging.warning(
                    # This warning was included with the original BertForMaskedLM.
                    f"If you want to use `{name_prefix}BertForMaskedLM` make sure "
                    " `config.is_decoder=False` for bi-directional self-attention."
                )

            self.bert = bert_cls(config, add_pooling_layer=False)
            self.cls = BertOnlyMLMHead(config)

            self.init_weights()

    # Rename new model class and format docstring.
    new_cls = NewBertForMaskedLM
    new_cls.__name__ = new_cls.__name__.replace("New", name_prefix)
    new_cls.__doc__ = new_cls.__doc__.format(name_prefix=name_prefix)

    return new_cls


def create_sequence_classification_class(bert_cls, name_prefix):
    """
    Create a BertForSequenceClassification that calls `bert_cls` in it's forward.
    """

    class NewBertForSequenceClassification(BertForSequenceClassification):
        """
        Sparse Bert Model transformer with a sequence classification/regression head on
        top (a linear layer on top of the pooled output) e.g. for GLUE tasks. Calls
        {name_prefix}BertModel in forward.

        This is adapted from `BertForMaskedLM`_

        .. BertForSequenceClassification:
            https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification
        """

        def __init__(self, config):
            # Call the init one parent class up.
            # Otherwise, the model will be defined twice.
            BertPreTrainedModel.__init__(self, config)

            self.num_labels = config.num_labels

            # Replace `BertModel` with SparseBertModel.
            self.bert = bert_cls(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            self.init_weights()

    # Rename new model class and format docstring.
    new_cls = NewBertForSequenceClassification
    new_cls.__name__ = new_cls.__name__.replace("New", name_prefix)
    new_cls.__doc__ = new_cls.__doc__.format(name_prefix=name_prefix)

    return new_cls


def create_question_answering_class(bert_cls, name_prefix):
    """
    Create a BertForQuestionAnswering that calls `bert_cls` in it's forward.
    """

    class NewBertForQuestionAnswering(BertForQuestionAnswering):
        """
        Bert Model with a span classification head on top for extractive
        question-answering tasks like SQuAD (a linear layers on top of the
        hidden-states output to compute `span start logits` and
        `span end logits`).

        https://huggingface.co/transformers/model_doc/bert.html#transformers.models.bert.modeling_bert.BertForQuestionAnswering  # noqa: E501
        """

        def __init__(self, config):

            # Call the init one parent class up.
            # Otherwise, the model will be defined twice.
            BertPreTrainedModel.__init__(self, config)

            self.do_beam_search = config.beam_searh

            if config.beam_search:
                self.start_n_top = config.start_n_top
                self.end_n_top = config.end_n_top

                self.transformer = bert_cls(config)
                self.start_logits = PoolerStartLogits(config)
                self.end_logits = PoolerEndLogits(config)
                self.answer_class = PoolerAnswerClass(config)

                self.init_weights()

            else:
                self.num_labels = config.num_labels

                # Replace `BertModel` with SparseBertModel.
                self.bert = bert_cls(config)
                self.dropout = nn.Dropout(config.hidden_dropout_prob)
                self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

                self.init_weights()

        def forward(many_arguments):

            # copied from https://huggingface.co/transformers/_modules/transformers/models/xlnet/modeling_xlnet.html#XLNetForQuestionAnsweringSimple
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            hidden_states = transformer_outputs[0]
            start_logits = self.start_logits(hidden_states, p_mask=p_mask)

            outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, let's remove the dimension added by batch splitting
                for x in (start_positions, end_positions, cls_index, is_impossible):
                    if x is not None and x.dim() > 1:
                        x.squeeze_(-1)

                # during training, compute the end logits based on the ground truth of the start position
                end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

                loss_fct = CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

                if cls_index is not None and is_impossible is not None:
                    # Predict answerability from the representation of CLS and START
                    cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                    loss_fct_cls = nn.BCEWithLogitsLoss()
                    cls_loss = loss_fct_cls(cls_logits, is_impossible)

                    # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                    total_loss += cls_loss * 0.5

                if not return_dict:
                    return (total_loss,) + transformer_outputs[1:]
                else:
                    return XLNetForQuestionAnsweringOutput(
                        loss=total_loss,
                        mems=transformer_outputs.mems,
                        hidden_states=transformer_outputs.hidden_states,
                        attentions=transformer_outputs.attentions,
                    )

            else:
                # during inference, compute the end logits based on beam search
                bsz, slen, hsz = hidden_states.size()
                start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

                start_top_log_probs, start_top_index = torch.topk(
                    start_log_probs, self.start_n_top, dim=-1
                )  # shape (bsz, start_n_top)
                start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
                start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
                start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

                hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                    start_states
                )  # shape (bsz, slen, start_n_top, hsz)
                p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
                end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
                end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

                end_top_log_probs, end_top_index = torch.topk(
                    end_log_probs, self.end_n_top, dim=1
                )  # shape (bsz, end_n_top, start_n_top)
                end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
                end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

                start_states = torch.einsum(
                    "blh,bl->bh", hidden_states, start_log_probs
                )  # get the representation of START as weighted sum of hidden states
                cls_logits = self.answer_class(
                    hidden_states, start_states=start_states, cls_index=cls_index
                )  # Shape (batch size,): one single `cls_logits` for each sample

                if not return_dict:
                    outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
                    return outputs + transformer_outputs[1:]
                else:
                    return XLNetForQuestionAnsweringOutput(
                        start_top_log_probs=start_top_log_probs,
                        start_top_index=start_top_index,
                        end_top_log_probs=end_top_log_probs,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                        mems=transformer_outputs.mems,
                        hidden_states=transformer_outputs.hidden_states,
                        attentions=transformer_outputs.attentions,
                    )

    # Rename new model class and format docstring.
    new_cls = NewBertForQuestionAnswering
    new_cls.__name__ = new_cls.__name__.replace("New", name_prefix)
    new_cls.__doc__ = new_cls.__doc__.format(name_prefix=name_prefix)

    return new_cls
