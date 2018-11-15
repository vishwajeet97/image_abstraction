import cv2
import numpy as np
class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def ETF(self):
		def clip(x, lo, hi):
			return max(lo, min(x, hi))

		gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
		sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
		mag = np.sqrt(sobelx**2 + sobely**2)
		mag = cv2.normalize(mag, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

		KERNEL_SIZE = 5
		KSby2 = int((KERNEL_SIZE - 1)/2)
		H, W = self.image.shape

		vector_field = np.stack([sobelx, sobely], axis=2)
		updated_vf = np.zeros(vector_field.shape)

		for i in range(self.image.shape[0]):
			for j in range(self.image.shape[1]):
				min_x, max_x = max(i-KSby2,0), min(i+KSby2+1, H)
				min_y, max_y = max(j-KSby2,0), min(j+KSby2+1, W)
				 
				patch = vector_field[min_x:max_x, min_y:max_y]
				patch_mag = mag[min_x:max_x, min_y:max_y]
				


	def run(self):
		etf = self.ETF()
		return etf


