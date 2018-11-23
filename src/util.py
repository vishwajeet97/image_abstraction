import numpy as np
import matplotlib.pyplot as plt
from src.vectorplot import lic_internal, kernels
import cv2
import pylab as plt

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

def view_vf(vf, ite):
	texture = np.random.rand(vf.shape[0],vf.shape[1]).astype(np.float32)
	kernellen=31
	kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
	kernel = kernel.astype(np.float32)
	# print(vf.shape)
	x = vf[:,:,[0]].astype(np.float32)
	x = np.squeeze(x, axis=2)
	y = vf[:,:,[1]].astype(np.float32)
	y = np.squeeze(y, axis=2)
	# print(x.shape)
	# print(y.shape)
	visual_vf = lic_internal.line_integral_convolution(x, y, texture, kernel)
	# print(np.sum(np.where(visual_vf>0,1,0)))
	# cv2.imshow('image1',np.array(visual_vf, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	plt.bone()
	plt.clf()
	plt.axis('off')
	plt.figimage(visual_vf)
	dpi = 100
	plt.gcf().set_size_inches((vf.shape[1]/float(dpi),vf.shape[0]/float(dpi)))
	plt.savefig("etf_iter"+str(ite)+".png",dpi=dpi) 

def view_magvf(vf):
	cv2.imshow('image1',np.linalg.norm(vf,axis=2))
	cv2.waitKey(0)
	cv2.destroyAllWindows()