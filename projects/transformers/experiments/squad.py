#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Finetuning on Squad, not Glue.
"""

from copy import deepcopy

from transformers import Trainer

from callbacks import TrackEvalMetrics
from trainer_mixins import QuestionAnsweringMixin

from .finetuning import finetuning_bert_100k_glue_get_info


class QuestionAnsweringTrainer(QuestionAnsweringMixin, Trainer):
    pass


debug_bert_squad_base = deepcopy(finetuning_bert_100k_glue_get_info)
debug_bert_squad_base.update(
    # Model Args
    model_name_or_path="bert-base-cased",
    finetuning=True,
    task_names=None,
    task_name="squad",
    dataset_name="squad",
    dataset_config_name="plain_text",
    trainer_class=QuestionAnsweringTrainer,
    max_seq_length=128,
    do_train=True,
    do_eval=True,
    do_predict=False,
    trainer_callbacks=[
        TrackEvalMetrics(),
    ],
    max_steps=100,
    eval_steps=20,
    rm_checkpoints=True,
    load_best_model_at_end=True,
    warmup_ratio=0.
)

# Supposed to train in about 24 minutes, takes an hour though
# Expect f1 score of 88.52, exact_match of 81.22
bert_squad_replication = deepcopy(debug_bert_squad_base)
bert_squad_replication.update(
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    max_seq_length=384,
    doc_stride=128,
    learning_rate=3e-5,
    save_steps=1_000,
    eval_steps=1_000,
    logging_steps=100_000,  # intended to not log until the end
)

del bert_squad_replication["max_steps"]

debug_squad_v1_no_beam = deepcopy(bert_squad_replication)
debug_squad_v1_no_beam.update(
    save_steps=100,
    eval_steps=100,
    max_steps=500,
)


# Neural magic trains for 30 epochs on squad which is ~90K training samples
# = 2.7M samples. Our models are pretrained with 100k steps with a batch size
# of 8 = 800K samples. 2.7M - 800K ~ 1.9M, 1.9M // batch size of 12 = 
# a budget of~ 150K training steps allowed on squad for total training steps
# to be similar
bert_100k_squad = deepcopy(bert_squad_replication)
bert_100k_squad.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",
    trainer_callbacks=[
        TrackEvalMetrics()],
    max_steps=150_000,  # just over 20 epochs
    eval_steps=10_000,
    save_steps=10_000,
    logging_steps=10_000,
)


# Export configurations in this file
CONFIGS = dict(
    debug_bert_squad_base=debug_bert_squad_base,
    bert_squad_replication=bert_squad_replication,
    debug_squad_v1_no_beam=debug_squad_v1_no_beam,
    bert_100k_squad=bert_100k_squad,
)
