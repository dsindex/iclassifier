from __future__ import absolute_import, division, print_function

# ------------------------------------------------------------------------------ #
# reference
#   https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# ------------------------------------------------------------------------------ #

import sys
import io
import os
import argparse
import json
import time
import pdb
import logging

import torch
import onnx
import onnxruntime

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--onnx_model_path', type=str, default='super_resolution.onnx')
    opt = parser.parse_args()

    # validate
    onnx_model = onnx.load(opt.onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # load and run
    ort_session = onnxruntime.InferenceSession(opt.onnx_model_path)

    # input to the model
    batch_size = 1
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
   
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

if __name__ == '__main__':
    main()
