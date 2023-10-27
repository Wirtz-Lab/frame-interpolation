import numpy as np
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None

def reshape_split(image:np.ndarray,kernel_size:tuple):
    img_height,img_width,channels=image.shape
    tile_height,tile_width = kernel_size
    tiled_array = image.reshape(img_height//tile_height,
                                tile_height,
                                img_width//tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array

if __name__=="__main__":
    src = r'\\10.99.68.178\Saurabh\CODA methods lungs\10x\WSI\registeredE_scale\cropped'

    imlist = [_ for _ in os.listdir(src) if _.endswith('tif')]
    dst = os.path.join(src,'tiledwsi')
    if not os.path.exists(dst): os.mkdir(dst)
    for im in imlist:
        imobj = Image.open(os.path.join(src,im))
        # Image to Array
        imnp = np.array(imobj)
        # Delete Image to save memory (optional)
        imobj.close()
        # Get wsi shape
        h,w,_=imnp.shape
        # Define tile size
        tile_height, tile_width = (1024,1024)
        # Pad wsi to generate equal size tiles on the edges as the rest
        imnpr = np.pad(imnp, pad_width=[(0, tile_height-h%tile_height),(0, tile_width-w%tile_width),(0, 0)], mode='constant', constant_values=0)
        # Get padded wsi shape
        h2,w2,_=imnpr.shape
        # Tile wsi
        tiles = reshape_split(imnpr, (1024,1024))
        a,b,_,_,_= [_ for _ in tiles.shape]
        imn,_ = os.path.splitext(im)
        for i in range(a):
            for j in range(b):
                Image.fromarray(tiles[i,j,:,:,:]).save(os.path.join(dst,'{}_{}_{}_1024x1024x3.tif'.format(imn,i,j)))

