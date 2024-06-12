# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:36:55 2024

@author: ramad
"""

import splitfolders  # or import split_folders

input_folder = 'D:/Intelligent Multimedia Network/Research/Riset Pak Dedi/Dataset Olah/Dataset Olah Pertama/'
output_folder = 'D:/Intelligent Multimedia Network/Research/Riset Pak Dedi/Dataset Olah/Dataset Olah Kedua/Citra Asli/'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.90, .10), group_prefix=None) # default values