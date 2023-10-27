import os
from PIL import Image
from tqdm import tqdm

def format_pwu2triplet(src,dst,imormsk,skip):
    if not os.path.exists(dst): os.mkdir(dst)
    dstseq = os.path.join(dst,'sequences')
    if not os.path.exists(dstseq): os.mkdir(dstseq)
    fols = [_ for _ in os.listdir(src) if os.path.isdir(os.path.join(src, _))]
    tri_testlist = os.path.join(dst, 'sequences.txt')
    file = open(tri_testlist, 'w')
    for fol in tqdm(fols):
        ims = [_ for _ in os.listdir(os.path.join(*[src,fol,'original_frames'])) if _.endswith(('.png','.jpg','tif'))]
        dstroi = os.path.join(dstseq,fol)
        if not os.path.exists(dstroi): os.mkdir(dstroi)
        for idx, triplet in enumerate(zip(ims[::skip+2],ims[int((skip+1)/2)::skip+2],ims[skip+1::skip+2])):
            zidx = str(idx).zfill(3)
            file.write(fol + '/' + zidx + "\n")
            dsttri = os.path.join(dstroi,zidx)
            if not os.path.exists(dsttri): os.mkdir(dsttri)
            for idxtri,im in enumerate(triplet):
                imsrc = os.path.join(*[src,fol,imormsk,im])
                imdst = os.path.join(dsttri,'im'+str(idxtri+1)+'.png')
                imobj = Image.open(imsrc)
                # if imormsk == 'original_frames': imobj = Image.merge('RGB', (imobj, imobj, imobj))
                imobj.save(imdst)
    file.close()

if __name__ == '__main__':
    #src = r'\\10.99.68.178\Saurabh\Andre_CODA_fallopian_tube\10x_python\registeredE\cropped\Authentic_tiles'
    #dst = r'\\10.99.68.178\Saurabh\Andre_CODA_fallopian_tube\10x_python\registeredE\cropped\Trainingset_Skip3'
    src = r'\\10.99.68.178\Saurabh\film_train_v2\sequences\compiled_test\liver'
    dst = r'\\10.99.68.178\Saurabh\film_train_v2\sequences\trainingset_sk3_liv\test'
    skip=3
    imormsk='original_frames'
    format_pwu2triplet(src, dst,imormsk,skip)
