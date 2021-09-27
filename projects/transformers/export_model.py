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

"""
Pretrained models need to be exported to be used for finetuning.
Only required argument for this script is the checkpoint folder.

Not tested for modified sparse models.
"""

import argparse
import os

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)

# Import models. This will update Transformer's model mappings so that custom models can
# be loaded via AutoModelForMaskedLM.
import models  # noqa F401


def save_model(model, destination_folder, model_name):
    """
    Core function that saves models
    """

    print("Saving with model type:", model.__class__.__name__)
    destination_file_path = os.path.join(destination_folder, model_name)
    model.save_pretrained(destination_file_path)
    print(f"Model saved at {destination_file_path}")


def export_model(checkpoint_folder, destination_folder, model_name, model_type):
    if not model_name:
        model_name = os.path.split(checkpoint_folder)[-1]

    if model_type == "PRETRAINING":
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_folder)
    elif model_type == "GLUE":
        model = AutoModelForSequenceClassification(checkpoint_folder)
    elif model_type == "SQUAD":
        model = AutoModelForQuestionAnswering(checkpoint_folder)
    else:
        print(f"Unknown model type specified: {model_type}")
        print("model_type must be one of [PRETRAINING, GLUE, SQUAD]")
        raise ValueError

    save_model(model, destination_folder, model_name)


def export_all_glue(parent_folder, destination_folder, model_name, model_type):
    """
    Simply loops over export_model once for each finetuning task.

    This function is for convenience, so you can run this script once for all 9 GLUE
    tasks, instead of 9 separate times. Note, this assumes there is a single run_*
    directory under each task directory with the best model saved there. If a run broke
    for some reason and the results for a single pretrained model are distributed under
    multiple configs (e.g. trifecta_85_hpchase, trifecta_85_hpchase_follow_up), this
    will need to be called once per config.
    """

    msg_1 = f"model_type must be GLUE, but you specified {model_type}"
    assert model_type == "GLUE", msg_1

    # Get all task subdirectories from parent folder
    sub_dirs = os.listdir(parent_folder)
    task_dirs = []
    for sub_dir in sub_dirs:
        full_dir = os.path.join(parent_folder, sub_dir)
        if os.path.isdir(full_dir):
            task_dirs.append(full_dir)

    # Ensure there is only one run subdirectory per task, and export one
    # model for each task
    for task_dir in task_dirs:
        task_sub_dirs = os.listdir(task_dir)
        run_dirs = []
        for task_sub_dir in task_sub_dirs:
            run_full_dir = os.path.join(task_dir, task_sub_dir)
            if os.path.isdir(task_sub_dir):
                run_dirs.append(run_full_dir)

        msg_2 = "This function assumes there is a single run directory per task. "
        msg_2 += f"but the contents of this directory inlcude {run_dirs}"
        assert len(run_dirs) == 1, msg_2
        run_dir = run_dirs[0]

        task_model_name = "_".join(model_name, task_dir)
        export_model(run_dir, destination_folder, task_model_name, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_folder", type=str,
                        help="Path to checkpoint to convert")
    parser.add_argument("--destination_folder", type=str,
                        default="/mnt/efs/results/pretrained-models/transformers-local",
                        help="Where to save the converted model")
    parser.add_argument("--model_name", type=str,
                        default=None,
                        help="Name of the model to save")
    parser.add_argument("--model_type", type=str, default="PRETRAINING",
                        choices=["PRETRAINING", "GLUE", "SQUAD"],
                        help="Save a pretraining model (AutoModelForMaskedLM) or "
                             "a GLUE model (AutoModelForSequenceClassification) "
                             "or a SQUAD model (AutoModelForQuestionAnswering)")
    parser.add_argument("--all_glue", type=bool, default=False,
                        help="If True, checkpoint folder should point to a parent "
                             "directory, with subdirectories for each task. all-glue "
                             "indicates that this will call ")
    args = parser.parse_args()

    # export a model for each available finetuning task
    if args.all_glue:
        export_all_glue(**args.__dict__)
    # save a single model
    else:
        export_model(**args.__dict__)
