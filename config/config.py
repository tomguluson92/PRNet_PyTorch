# -*- coding: utf-8 -*-

"""
    @date: 2019.11.13
    @function: hyparameter for training & inference.
"""

FLAGS = {"start_epoch": 0,
         "target_epoch": 500,
         "device": "cuda",
         "mask_path": "./utils/uv_data/uv_weight_mask_gdh.png",
         "lr": 0.0001,
         "batch_size": 16,
         "save_interval": 5,
         "normalize_mean": [0.485, 0.456, 0.406],
         "normalize_std": [0.229, 0.224, 0.225],
         "images": "./results",
         "gauss_kernel": "original",
         "summary_path": "./prnet_runs",
         "summary_step": 0,
         "resume": True}