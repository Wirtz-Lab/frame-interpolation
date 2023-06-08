import shutil
import os,pathlib

def triplet2pair(src,dstfn):
    src = pathlib.Path(src)
    srcfn = os.path.basename(src)
    # create directories
    [os.mkdir(str(_).replace(srcfn,dstfn)) for _ in src.glob('**')]
    #copy im1 and im3 files
    [shutil.copy(str(_),str(_).replace(srcfn,dstfn)) for _ in src.glob('**/*/*.png') if 'im2' not in str(_)]

if __name__ == '__main__':
    src = r'\\shelter\Kyu\motility_interpolation\testv1'
    dstfn = 'testv1_pair'
    triplet2pair(src, dstfn)