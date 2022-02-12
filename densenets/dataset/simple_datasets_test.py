# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% language="javascript"
# IPython.OutputArea.auto_scroll_threshold = 9999;

# %%
import sys
sys.path
sys.path.append('../../modules')

# %%
from spartan_dataset_masked import SpartanDataset
import dense_correspondence_manipulation.utils.utils as utils

utils.add_dense_correspondence_to_python_path()
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType

import os
import torch
import numpy as np
# %matplotlib inline

dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'caterpillar_upright.yaml')

dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)

dataset = SpartanDataset(debug=True, config=dataset_config)

# %%
dataset_config

# %%
match_type, image_a_rgb, image_b_rgb, \
matches_a, matches_b, masked_non_matches_a, \
masked_non_matches_a, non_masked_non_matches_a, \
non_masked_non_matches_b, blind_non_matches_a, \
blind_non_matches_b, metadata = dataset.get_single_object_within_scene_data()

# %%
