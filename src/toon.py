import cv2
import numpy as np
import lic.lic_internal

class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def ETF(self):
		gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
		sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
		mag = np.sqrt(sobelx**2 + sobely**2)
		KERNEL_SIZE = 5
		texture = np.random.rand(sobelx.shape(0),sobelx.shape(1)).astype(np.float32)
		kernellen=31
		kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
		kernel = kernel.astype(np.float32)
		lic_internal.line_integral_convolution(np.stack(sobelx,sobely,axis=2), texture, kernel)
		return sobely

	def run(self):
		etf = self.ETF()
		return etf


