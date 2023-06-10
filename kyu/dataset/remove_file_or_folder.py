import shutil,os
import tensorflow as tf

[os.remove(_) for _  in tf.io.gfile.glob(r'\\shelter\Kyu\motility_interpolation\filmtest_train_skip3\sequences\*\*\*og*') if os.path.isfile(_)]
[shutil.rmtree(_) for _ in tf.io.gfile.glob(r'\\shelter\Kyu\motility_interpolation\filmtest_train_skip3\sequences\*\*\og*')  if os.path.isdir(_)]