import numpy as np
from src.lic.lic_internal import line_integral_convolution
import cv2

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = nrows,ncols,BSZ[0],BSZ[1]
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(nrows, ncols, BSZ[0]*BSZ[1])[:,::stepsize]

def view_vf(vf):
	texture = np.random.rand(vf.shape[1],vf.shape[0]).astype(np.float32)
	kernellen=31
	kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
	kernel = kernel.astype(np.float32)
	print(kernel.shape)
	visual_vf = line_integral_convolution(vf, texture, kernel)
	print(np.sum(np.where(visual_vf>0,1,0)))
	cv2.imshow('image1',visual_vf)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
