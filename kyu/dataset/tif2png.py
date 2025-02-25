from PIL import Image
import os
import glob
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import cv2
def tif2png(tifpth,dst):
    imnm = tifpth.replace('tif','png')
    fol,imnm = os.path.split(imnm)
    im = Image.open(tifpth)
    im = np.array(im)

    scale_percent = 50 # percent of original size
    width = round(im.shape[1] * scale_percent / 100)
    height = round(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA) #inter_area for downsizing

    Image.fromarray(im).save(os.path.join(dst,imnm))

if __name__=='__main__':
    src = r'\\shelter\Pratik\KYU\Raw Images\GT22\12HOUR\REP2'
    ims = glob.glob(os.path.join(src, '*tif'))
    ims = natsorted(ims)
    dst1 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT22_12HR_Nuclei_50percent'
    dst2 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT22_12HR_Actin_50percent'
    if not os.path.exists(dst1): os.mkdir(dst1)
    if not os.path.exists(dst2): os.mkdir(dst2)
    ims_tmpc1 = [tif2png(_, dst1) for _ in tqdm(ims) if "ch00" in _]
    ims_tmpc2 = [tif2png(_, dst2) for _ in tqdm(ims) if "ch01" in _]
    #
    # src = r'\\shelter\Pratik\KYU\Raw Images\GT22\12HOUR\REP2'
    # ims = glob.glob(os.path.join(src, '*tif'))
    # ims = natsorted(ims)
    # dst1 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT22_12HR_Nuclei'
    # dst2 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT22_12HR_Actin'
    # if not os.path.exists(dst1): os.mkdir(dst1)
    # if not os.path.exists(dst2): os.mkdir(dst2)
    # ims_tmpc1 = [tif2png(_, dst1) for _ in tqdm(ims) if "ch00" in _]
    # ims_tmpc2 = [tif2png(_, dst2) for _ in tqdm(ims) if "ch01" in _]
    #
    # src = r'\\shelter\Pratik\KYU\Raw Images\GT125\12HOUR\REP2'
    # ims = glob.glob(os.path.join(src, '*tif'))
    # ims = natsorted(ims)
    # dst1 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT125_12HR_Nuclei'
    # dst2 = r'\\shelter\Kyu\motility_interpolation\dataset\original_frames\GT125_12HR_Actin'
    # if not os.path.exists(dst1): os.mkdir(dst1)
    # if not os.path.exists(dst2): os.mkdir(dst2)
    # ims_tmpc1 = [tif2png(_, dst1) for _ in tqdm(ims) if "ch00" in _]
    # ims_tmpc2 = [tif2png(_, dst2) for _ in tqdm(ims) if "ch01" in _]