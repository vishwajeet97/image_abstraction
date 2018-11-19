import cv2
import numpy as np
import src.util as util

class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def ETF(self):
		def clip(x, lo, hi):
			return max(lo, min(x, hi))
		smoothen_image = cv2.GaussianBlur(self.image, (5,5), 0, 0, cv2.BORDER_DEFAULT)
		gray_img = cv2.cvtColor(smoothen_image, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray_img,cv2.CV_32F,1,0,ksize=3)
		sobely = cv2.Sobel(gray_img,cv2.CV_32F,0,1,ksize=3)

		mag, angle = cv2.cartToPolar(sobelx, sobely, magnitude=None, angle=None, angleInDegrees=True)
		mag = cv2.normalize(mag, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
		angle = (angle + 90) % 360
		x, y = cv2.polarToCart(mag, angle, x=None, y=None, angleInDegrees=True)
		vector_field = np.stack([x, y], axis=2)

		KERNEL_SIZE = 9
		KSby2 = int((KERNEL_SIZE - 1)/2)
		H, W, C = self.image.shape

		indices = np.indices([H, W]).transpose((1,2,0))

		ite = 0
		util.view_vf(vector_field,ite)
		ite += 1

		for iter_no in range(3):
			updated_vf = np.zeros(vector_field.shape)
			for i in range(self.image.shape[0]):
				for j in range(self.image.shape[1]):
					min_x, max_x = max(i-KSby2,0), min(i+KSby2+1, H)
					min_y, max_y = max(j-KSby2,0), min(j+KSby2+1, W)
					patch = vector_field[min_x:max_x, min_y:max_y]
					patch_mag = mag[min_x:max_x, min_y:max_y]
					patch_indices = indices[min_x:max_x, min_y:max_y, :]

					ws = np.linalg.norm(patch_indices - indices[i, j, :], axis=2)
					ws = np.where(ws > self.config.etf_radius, 0.0, 1.0)
					wd = np.einsum('ijk,k->ij', patch, vector_field[i][j])
					wm = (patch_mag - mag[i][j] + 1)/2

					filt = np.multiply(np.multiply(ws, wd), wm)
					updated_vf[i][j] = np.sum(np.einsum('ijk,ij->ijk', patch, filt), axis=(0,1))
					norm = np.linalg.norm(updated_vf[i][j])
					if norm != 0:
						updated_vf[i][j] = updated_vf[i][j]/norm
			# print(updated_vf)
			util.view_vf(updated_vf,ite)
			vector_field = updated_vf
			ite += 1

		return updated_vf

	def run(self):
		etf = self.ETF()
		return etf


