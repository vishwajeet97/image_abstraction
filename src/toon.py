import cv2
import numpy as np
import src.util as util
import math
import pylab as plt

class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def gaussian(self, point, sigma): # eqn (7) in paper #TODO:use some librart
		# print(point)
		# print(sigma)
		return math.exp(-(point**2)/(2*(sigma**2)))/(math.sqrt(2*math.pi*sigma))

	def edge_dog(self, t, sigma_c, p): # eqn (8) in paper
		return self.gaussian(t, sigma_c) - p * self.gaussian(t, 1.05*sigma_c)

	def getPointsOnLine(self, slope, size): #1X1 3X3 5X5 7X7
		points = np.zeros((size,2))
		x = 0
		y = 0
		points[0] =[0,0]
		i = 1
		if abs(slope)<=1:
			for x in range(size//2):
				x += 1
				y += slope
				points[i][0] = x
				points[i][1] = math.floor(y+0.5)
				points[size-i][0] = -points[i][0]
				points[size-i][1] = -points[i][1]
				i +=1
		else:
			slope_ = 1/slope
			for y in range(size//2):
				x += slope_
				y += 1
				points[i][0] = math.floor(x+0.5)
				points[i][1] = y
				points[size-i][0] = -points[i][0]
				points[size-i][1] = -points[i][1]
				i +=1

		return points

	def getF(self,size): #1X1 3X3 5X5 7X7
		sigma_c = self.config.sigma_c
		p = self.config.rho
		points = np.zeros((size))
		points[0] = self.edge_dog(0,sigma_c,p)
		for i in range(1, size//2 + 1):
			# print(i)
			points[i] = self.edge_dog(i,sigma_c,p); #repititive calc TODO
			points[size-i] = points[i]
		return points

	def getG(self,size): #1X1 3X3 5X5 7X7
		sigma_m = self.config.sigma_m
		points = np.zeros((size))
		points[0] = self.gaussian(0,sigma_m)
		for i in range(1, size//2 + 1):
			points[i] = self.gaussian(i,sigma_m); #repititive calc TODO
			points[size-i] = points[i]
		return points

	def FDOG(self):
		# fdog = np.copy(self.preProcess/255) #cloning problem?
		img = np.float64(self.preProcess)/255
		fdog = np.copy(img)
		updated_fdog = np.zeros(fdog.shape)
		M = self.image.shape[0]
		N = self.image.shape[1]
		for iter_no in range(self.config.fdog_iterations):
			Hg = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					t0 = self.etf[i][j]
					direc = math.atan2(t0[0],-t0[1]);
					kernel_size = 7 #7? need to fix the size acc. to sigma_c
					points = self.getPointsOnLine(direc, kernel_size).astype(np.int)
					f = self.getF(kernel_size)
					# print(f)
					ty = [1 if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					I = [fdog[i+p[0]][j+p[1]] if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					Hg[i][j] = np.sum(I*f)/np.sum(f*ty)
					if(Hg[i][j]) < 0:
						print("YES")
			print(np.sum(np.array(Hg) < 0))
			He = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					t0 = self.etf[i][j]
					direc = math.atan2(t0[1],t0[0]);
					kernel_size = 7 #7? need to fix the size acc. to sigma_c
					points = self.getPointsOnLine(direc, kernel_size).astype(np.int)
					g = self.getG(kernel_size)
					ty = [1 if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					H = [Hg[i+p[0]][j+p[1]] if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					val = np.sum(H*g)/np.sum(g*ty)
					# print(val)
					if val<0 and 1+np.tanh(val) < self.config.fdog_threshold:
						He[i][j] = 0
					else:
						He[i][j] = 255
			print(np.sum(np.array(He) > 0))
			updated_fdog = He
			fdog = np.minimum(updated_fdog, img)
		# cv2.imshow('image',updated_fdog)
		# # cv2.imshow('abstract_image',abstract_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		plt.bone()
		plt.clf()
		plt.axis('off')
		plt.figimage(updated_fdog)
		dpi = 100
		print(updated_fdog.shape)
		plt.gcf().set_size_inches((updated_fdog.shape[1]/float(dpi),updated_fdog.shape[0]/float(dpi)))
		plt.savefig("fdog_iter"+str(1)+".png",dpi=dpi) 

		self.fdog = updated_fdog

	def ETF(self):
		smoothen_image = cv2.GaussianBlur(self.image, (5,5), 0, 0, cv2.BORDER_DEFAULT)
		gray_img = cv2.cvtColor(smoothen_image, cv2.COLOR_BGR2GRAY)
		self.preProcess = gray_img
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

		updated_vf = np.zeros(vector_field.shape)
		for iter_no in range(self.config.etf_iterations):
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

		self.etf = updated_vf

	def run(self):
		self.ETF()
		self.FDOG()
		return self.etf


