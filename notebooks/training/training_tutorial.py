# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import logging
import os

import densenets.dense_correspondence_manipulation.utils.utils as utils
import torch
from densenets.dataset.spartan_dataset_masked import SpartanDataset
from densenets.evaluation.evaluation import DenseCorrespondenceEvaluation
from densenets.training.training import DenseCorrespondenceTraining

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Load the configuration for training

# %%
config_filename = os.path.join(
    utils.getDenseCorrespondenceSourceDir(),
    'config',
    'dense_correspondence',
    'dataset',
    'composite',
    'caterpillar_upright.yaml',
)
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(
    utils.getDenseCorrespondenceSourceDir(),
    'config',
    'dense_correspondence',
    'training',
    'training.yaml',
)

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "trained_models/tutorials"
num_iterations = 3500
d = 3  # the descriptor dimension
name = "caterpillar_%d" % (d)
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations

TRAIN = True
EVALUATE = True

# %% [markdown]
# ## Train the network
#
# This should take about ~12-15 minutes with a GTX 1080 Ti


# %%
class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.layer = torch.nn.Linear(1, 10)

    def forward(self, x):
        return self.layer(x)


# %%
t = Test()

# %%
t.cuda()

# %%
t(torch.rand([20, 1]).cuda())

# %%
# All of the saved data for this network will be located in the
# code/data/pdc/trained_models/tutorials/caterpillar_3 folder

if TRAIN:
    print(f"training descriptor of dimension {d}")
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
    print(f"finished training descriptor of dimension {d}")


# %% [markdown]
# ## Evaluate the network quantitatively
#
# This should take ~5 minutes.

# %%
model_folder = os.path.join(logging_dir, name)
model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)

if EVALUATE:
    DCE = DenseCorrespondenceEvaluation
    num_image_pairs = 100
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs)

# %% [markdown]
# See `evaluation_quantitative_tutorial.ipynb` for a better place to display the plots.
