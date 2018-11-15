import cv2
import numpy as np
class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def ETF(self):
		gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
		sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
		mag = np.sqrt(sobelx**2 + sobely**2)
		scale_factor = np.max(mag)/255
		mag = (mag/scale_factor).astype(np.uint8)
		KERNEL_SIZE = 5
		return sobely

	def run(self):
		etf = self.ETF()
		return etf


