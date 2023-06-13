import tensorflow as tf
import os
from natsort import natsorted

if __name__ == '__main__':
    src = r'\\fatherserverdw\Saurabh\Saurabh\Pancreas_Ashley_Files\test_tiles_triplet_v2'
    ims = tf.io.gfile.glob(f'{src}/sequences/*/*')
    tri_testlist = os.path.join(src, 'sequences.txt')
    file = open(tri_testlist, 'w')
    for line in natsorted([os.path.join(*os.path.normpath(_).split(os.sep)[-2:]) for _ in ims]):
        file.write(line + "\n")
    file.close()

