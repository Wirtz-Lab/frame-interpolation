import functools
import os
from typing import List

from eval import interpolator as interpolator_lib
from eval import util
from absl import logging
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def _output_frames(frames: List[np.ndarray], frames_dir: str):
  for idx, frame in tqdm(
      enumerate(frames), total=len(frames), ncols=100, colour='green'):
    util.write_image(f'{frames_dir}/im{idx:01d}.png', frame)
  logging.info('Output frames saved in %s.', frames_dir)

class ProcessDirectory(beam.DoFn):
  """DoFn for running the interpolator on a single directory at the time."""

  def setup(self):
    self.interpolator = interpolator_lib.Interpolator(
        modelpath, align,
        [bh, bw])

    if outputvideo:
      ffmpeg_path = util.get_ffmpeg_path()
      media.set_ffmpeg(ffmpeg_path)

  def process(self, directory: str):
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*{ext}'))
        for ext in input_ext
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    logging.info('Generating in-between frames for %s.', directory)
    frames = list(
        util.interpolate_recursively_from_files(
            input_frames, time2inter, self.interpolator))
    _output_frames(frames, f'{directory}/{modelname}_skip{len(frames)-2}')

    if outputvideo:
      media.write_video(f'{directory}/interpolated.mp4', frames, fps=fps)
      logging.info('Out;put video saved at %s/interpolated.mp4.', directory)


if __name__=='__main__':
    pattern = r"\\shelter\Kyu\motility_interpolation\filmtest_train_skip3\sequences\*\*"
    directories = tf.io.gfile.glob(pattern)
    # directories = directories[:2]
    print(directories)

    modelpath=r"\\shelter\Kyu\motility_interpolation\pretrained_models\film_net\Style\saved_model"
    # modelname=os.path.basename(modelpath)
    modelname='ogfilm'
    time2inter=2 #[1,3,7,15,31,...]

    align=64
    bh=1
    bw=1
    outputvideo=False
    fps=30
    input_ext=['1.png','3.png']

    pipeline = beam.Pipeline('DirectRunner')
    (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
       | 'Process directories' >> beam.ParDo(ProcessDirectory()))
    result = pipeline.run()
    result.wait_until_finish()