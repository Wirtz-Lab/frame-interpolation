import os
from shutil import copyfile
from PIL import Image
from tqdm import tqdm

def format_pwu2triplet(src,dst):
    dstseq = os.path.join(dst,'sequences')
    if not os.path.exists(dstseq): os.mkdir(dstseq)
    fols = [_ for _ in os.listdir(src) if os.path.isdir(os.path.join(src, _))]
    fols = fols
    tri_testlist = os.path.join(dst, 'sequences.txt')
    file = open(tri_testlist, 'w')
    for fol in tqdm(fols):
        ims = [_ for _ in os.listdir(os.path.join(*[src,fol,'original_frames'])) if _.endswith(('.png','.jpg','tif'))]
        dstroi = os.path.join(dstseq,fol)
        if not os.path.exists(dstroi): os.mkdir(dstroi)
        for idx,triplet in enumerate(zip(ims[::3],ims[1::3],ims[2::3])):
            zidx = str(idx).zfill(3)
            file.write(fol + '/' + zidx + "\n")
            dsttri = os.path.join(dstroi,zidx)
            if not os.path.exists(dsttri): os.mkdir(dsttri)
            for idxtri,im in enumerate(triplet):
                # copyfile(os.path.join(*[src,fol,'original_frames',im]),os.path.join(dsttri,'im'+str(idxtri+1)+'.png'))
                imsrc = os.path.join(*[src,fol,'original_frames',im])
                imdst = os.path.join(dsttri,'im'+str(idxtri+1)+'.png')
                imobj = Image.open(imsrc)
                img = Image.merge('RGB', (imobj, imobj, imobj))
                img.save(imdst)
    file.close()

if __name__ == '__main__':
    src = r'\\shelter\Kyu\motility_interpolation\filmtest'
    dst = r'\\shelter\Kyu\motility_interpolation\filmtest_train'
    format_pwu2triplet(src, dst)