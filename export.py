#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config as cfg

from model.tensorpack_model import *

from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter

def export(args):
    model = AttentionOCR()
    print(model.get_inferene_tensor_names()[0])
    print(model.get_inferene_tensor_names()[1])
    predcfg = PredictConfig(
        model=model,
        session_init=SmartInit(args.checkpoint_path),
        input_names=model.get_inferene_tensor_names()[0],
        output_names=model.get_inferene_tensor_names()[1])

    # ModelExporter(predcfg).export_compact(args.pb_path, optimize=True, toco_compatible=True)
    # ModelExporter(predcfg).export_compact(args.pb_path, optimize=False)
    # ModelExporter(predcfg).export_compact(args.pb_path,optimize=False)
    ModelExporter(predcfg).export_serving(args.pb_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')

    parser.add_argument('--pb_path', type=str, help='path to save tensorflow pb model', default='serving')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model', default='/home/quannd/Vin-BigData/inceptionv4/inceptionv4/checkpoint_32*128/model-148200.data-00000-of-00001')

    args = parser.parse_args()
    export(args)
