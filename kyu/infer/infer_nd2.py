import os
from typing import List
from eval import interpolator as interpolator_lib
from eval import util
from absl import logging
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import nd2
import cv2
from time import time

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def _output_frames(frames: List[np.ndarray], frames_dir: str):
  if tf.io.gfile.isdir(frames_dir):
    old_frames = tf.io.gfile.glob(os.path.join(frames_dir,'frame_*.png'))
    if old_frames:
      logging.info('Removing existing frames from %s.', frames_dir)
      for old_frame in old_frames:
        tf.io.gfile.remove(old_frame)
  else:
    tf.io.gfile.makedirs(frames_dir)
  for idx, frame in tqdm(
      enumerate(frames), total=len(frames), ncols=100, colour='green'):
    util.write_image(os.path.join(frames_dir,'im{:01d}.png'.format(idx)), frame)
  print('Output frames saved in {}'.format(frames_dir))

def process(nd2pth,time2inter,rsf):
    tagname = os.path.splitext(os.path.basename(nd2pth))[0]
    # Calculate number of frames to skip
    skips = [1, 3, 7, 15, 31]
    skip = skips[time2inter - 1]
    print('skip:', skip)
    # Read nd2 file, split channel, rescale, make 3ch, apply inference
    nd2file = nd2.imread(nd2pth).astype(np.float32)
    pages, ch = [_ for _ in nd2file.shape][:2]
    sx, sy = [round(_ * rsf) for _ in nd2file.shape][-2:]
    nd2file = nd2file[::skip + 1]

    nd2file_c1 = nd2file[:, 0, :, :]
    _UINT16_MAX_F = float(np.iinfo(np.uint16).max)
    nd2file_c1 = nd2file_c1 / _UINT16_MAX_F
    nd2file_c1 = [cv2.resize(_, (sx, sy), interpolation=cv2.INTER_NEAREST) for _ in nd2file_c1]
    nd2file_c1 = [np.repeat(_[:, :, np.newaxis], 3, axis=2) for _ in nd2file_c1]
    frames = list(
        util.interpolate_recursively_from_memory(
            nd2file_c1, time2inter, interpolator))

    modelname = '{}_ogfilm_applyskip{}_c1'.format(tagname, skip)
    _output_frames(frames, os.path.join(nd2src, modelname))

    nd2file_c2 = nd2file[:, 1, :, :]
    nd2file_c2 = nd2file_c2 / _UINT16_MAX_F
    nd2file_c2 = [cv2.resize(_, (sx, sy), interpolation=cv2.INTER_NEAREST) for _ in nd2file_c2]
    nd2file_c2 = [np.repeat(_[:, :, np.newaxis], 3, axis=2) for _ in nd2file_c2]
    frames = list(
        util.interpolate_recursively_from_memory(
            nd2file_c2, time2inter, interpolator))

    modelname = '{}_ogfilm_applyskip{}_c2'.format(tagname, skip)
    _output_frames(frames, os.path.join(nd2src, modelname))

if __name__=='__main__':
    #Load model
    modelpath = r"\\shelter\Kyu\motility_interpolation\pretrained_models\film_net\Style\saved_model"
    interpolator = interpolator_lib.Interpolator(modelpath, 64, [1, 1])
    #Define skip number !Input!
    time2inter=1
    #Define rescale factor !Input!
    rsf=0.5
    #Load nd2 file !Input!
    # nd2src = r"\\motherserverdw\Lab Members\Praful\2023\20230824 2D cell migration fluorescence MDA HT MCF\Split\3. MCF7"
    rootsrc = r'\\motherserverdw\Lab Members\Praful\2023\20230824 2D cell migration fluorescence MDA HT MCF\Split'
    nd2srcs = [os.path.join(rootsrc,_) for _ in os.listdir(rootsrc) if os.path.isdir(os.path.join(rootsrc,_))]
    print(nd2srcs)
    for nd2src in nd2srcs:
        nd2s = [_ for _ in os.listdir(nd2src) if _.endswith('nd2')]
        print(nd2s)
        times2inter = [2,3,4,5]
        for time2inter in times2inter:
            start = time()
            [process(os.path.join(nd2src, _),time2inter,rsf) for _ in nd2s]
            print(round(time()-start),'sec elaspsed')


