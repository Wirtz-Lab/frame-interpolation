from eval import interpolator as interpolator_lib
from eval import util
import numpy as np
import os
import pathlib
from tqdm import tqdm

def infer_one_middle_frame(interpolator,triplet):
    image_1 = util.read_image(os.path.join(triplet,'im1.png'))
    image_batch_1 = np.expand_dims(image_1, axis=0)
    image_2 = util.read_image(os.path.join(triplet,'im3.png'))
    image_batch_2 = np.expand_dims(image_2, axis=0)
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]
    mid_frame_filepath = os.path.join(triplet,'im2_ogfilm.png')
    util.write_image(mid_frame_filepath, mid_frame)


if __name__=='__main__':
    model = r'\\shelter\Kyu\motility_interpolation\pretrained_models\film_net\Style\saved_model'
    interpolator = interpolator_lib.Interpolator(
      model_path= model,
      align=64,
      block_shape=[1,1])
    src = pathlib.Path(r'\\shelter\Kyu\motility_interpolation\filmtest_train_skip3')
    dirs = list(src.glob('sequences/*/*'))
    for triplet in tqdm(dirs): #1.2s/it
        # print(triplet)
        infer_one_middle_frame(interpolator,triplet)