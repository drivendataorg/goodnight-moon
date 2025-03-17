#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import shutil
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='./', type=str, help='Directory with a test dataset')
parser.add_argument('--assets_dir', default='assets_new', type=str, help='Directory with assets')
args = parser.parse_args()
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
# Copy first 2 folds of transcriber model whisper-medium.en
#------------------------------------------------------------------------------

os.makedirs(args.assets_dir, exist_ok=True)

for fold_id in range(2):
    weight = glob.glob('tuned_transcriber_0_mediumen/model-f%d-*' % fold_id)[0]
    dest = os.path.join(args.assets_dir, os.path.basename(weight).replace('model', 'trans%d' % fold_id))
    shutil.copy(weight, dest)
    print('Copied: %s -> %s' % (weight, dest))

#------------------------------------------------------------------------------
# Copy best cls models
#------------------------------------------------------------------------------

for model_id in range(6):
    weight = sorted(glob.glob('tuned_cls_%d_*/*.bin' % model_id), key=lambda x: float(x.split('-')[-3]))[0]
    dest = os.path.join(args.assets_dir, os.path.basename(weight).replace('model', 'cls%d' % model_id))
    shutil.copy(weight, dest)
    print('Copied: %s -> %s' % (weight, dest))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

