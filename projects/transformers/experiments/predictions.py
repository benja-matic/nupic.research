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

import os
import pathlib
from copy import deepcopy

from .hpchase import trifecta_80_hp_chase


# ---------
# Trifecta models
# ---------

finetuning_model_dir = "/mnt/efs/results/pretrained-models/transformers-local/finetuning"
trifecta_80_finetuning_model_dir = os.path.join(finetuning_model_dir, "trifecta_80")
trifecta_85_finetuning_model_dir = os.path.join(finetuning_model_dir, "trifecta_85")
trifecta_90_finetuning_model_dir = os.path.join(finetuning_model_dir, "trifecta_90")


trifecta_80_predict = deepcopy(trifecta_80_hp_chase)
trifecta_80_predict.update(
    task_name="glue",
    do_train=False,
    do_eval=False,
    do_predict=True,
    task_hyperparams=dict(
        cola=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "cola"]),
            do_eval=False,
        ),
        mrpc=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "mrpc"]),
            do_eval=False,
        ),
        mnli=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "mnli"]),
            do_eval=False,
        ),
        qnli=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "qnli"]),
            do_eval=False,
        ),
        qqp=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "qqp"]),
            do_eval=False,
        ),
        rte=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "rte"]),
            do_eval=False,
        ),
        sst2=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "sst2"]),
            do_eval=False,
        ),
        stsb=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "stsb"]),
            do_eval=False,
        ),
        wnli=dict(
            model_name_or_path="_".join([trifecta_80_finetuning_model_dir, "wnli"]),
            do_eval=False,
        ),
    )
)


trifecta_85_predict = deepcopy(trifecta_80_predict)
trifecta_85_predict.update(
    task_hyperparams=dict(
        cola=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "cola"]),
            do_eval=False,
        ),
        mrpc=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "mrpc"]),
            do_eval=False,
        ),
        mnli=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "mnli"]),
            do_eval=False,
        ),
        qnli=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "qnli"]),
            do_eval=False,
        ),
        qqp=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "qqp"]),
            do_eval=False,
        ),
        rte=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "rte"]),
            do_eval=False,
        ),
        sst2=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "sst2"]),
            do_eval=False,
        ),
        stsb=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "stsb"]),
            do_eval=False,
        ),
        wnli=dict(
            model_name_or_path="_".join([trifecta_85_finetuning_model_dir, "wnli"]),
            do_eval=False,
        ),
    )
)


trifecta_90_predict = deepcopy(trifecta_80_predict)
trifecta_90_predict.update(
    task_hyperparams=dict(
        cola=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "cola"]),
            do_eval=False,
        ),
        mrpc=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "mrpc"]),
            do_eval=False,
        ),
        mnli=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "mnli"]),
            do_eval=False,
        ),
        qnli=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "qnli"]),
            do_eval=False,
        ),
        qqp=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "qqp"]),
            do_eval=False,
        ),
        rte=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "rte"]),
            do_eval=False,
        ),
        sst2=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "sst2"]),
            do_eval=False,
        ),
        stsb=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "stsb"]),
            do_eval=False,
        ),
        wnli=dict(
            model_name_or_path="_".join([trifecta_90_finetuning_model_dir, "wnli"]),
            do_eval=False,
        ),
    )
)

CONFIGS=dict(
    trifecta_80_predict=trifecta_80_predict,
    trifecta_85_predict=trifecta_85_predict,
    trifecta_90_predict=trifecta_90_predict
)