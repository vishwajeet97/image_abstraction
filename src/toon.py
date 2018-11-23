import os
import math

import cv2
import numpy as np
import pylab as plt
import scipy.stats
from bresenham import bresenham

import src.util as util


class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config
		self.M, self.N, self.C = image.shape

	def gaussian(self, point, sigma): # eqn (7) in paper #TODO:use some librart
		# Update: can use scipy.stats.norm
		# https://stackoverflow.com/questions/12412895/calculate-probability-in-normal-distribution-given-mean-std-in-python/12413491
		return math.exp(-(point**2)/(2*(sigma**2)))/(math.sqrt(2*math.pi)*sigma)

	def edge_dog(self, t, sigma_c, p): # eqn (8) in paper
		return self.gaussian(t, sigma_c) - p * self.gaussian(t, 1.6*sigma_c)

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

	def getNextPointOnCurve(self, slope): #1X1 3X3 5X5 7X7
		point = np.zeros((2))
		if abs(slope)<=1:
			point[0] = 1
			point[1] = math.floor(slope+0.5)
		else:
			slope_ = 1/slope
			point[1] = 1
			point[0] = math.floor(slope_+0.5)
		return point

	def getF(self,size): #1X1 3X3 5X5 7X7
		sigma_c = self.config.sigma_c
		p = self.config.rho
		points = np.zeros((size))
		points[0] = self.edge_dog(0,sigma_c,p)
		for i in range(1, size//2 + 1):
			print(i)
			points[i] = self.edge_dog(i,sigma_c,p); #repititive calc TODO
			print(points[i])
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
		kernel_size = 7 #7? need to fix the size acc. to sigma_c
		f = self.getF(kernel_size)
		g = self.getG(kernel_size)
		print(f)
		print(g)
		for iter_no in range(self.config.fdog_iterations):
			Hg = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					t0 = self.etf[i][j]
					direc = math.atan2(t0[0],-t0[1]);
					points = self.getPointsOnLine(direc, kernel_size).astype(np.int)
					# print(f)
					ty = [1 if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					I = [fdog[i+p[0]][j+p[1]] if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					Hg[i][j] = np.sum(I*f)/np.sum(f*ty)
					# if(Hg[i][j]) < 0:
						# print("YES")
			print(np.sum(np.array(Hg) < 0))
			He = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					point = np.array([i,j])
					H = Hg[point[0]][point[1]]*g[0]
					G = g[0]
					for s in range(1,4):
						s_etf = self.etf[point[0]][point[1]]
						direc = math.atan2(s_etf[1],s_etf[0])
						p_delta = self.getNextPointOnCurve(direc).astype(np.int)
						point = point + p_delta
						if point[0]<M and point[0]>=0 and point[1]<N and point[1]>=0:
							H += Hg[point[0]][point[1]]*g[s]
							G += g[s]
						else:
							break
					point = np.array([i,j])
					for s in range(1,4):
						s_etf = self.etf[point[0]][point[1]]
						direc = math.atan2(-s_etf[1],-s_etf[0])
						p_delta = self.getNextPointOnCurve(direc).astype(np.int)
						point = point + p_delta
						if point[0]<M and point[0]>=0 and point[1]<N and point[1]>=0:
							H += Hg[point[0]][point[1]]*g[s]
							G += g[s]
						else:
							break
					val = H/G
					# print(val)
					if val<0 and 1+np.tanh(val) < self.config.fdog_threshold:
						He[i][j] = 0
					else:
						He[i][j] = 255
			print(np.sum(np.array(He) > 0))
			updated_fdog = He
			fdog = np.minimum(updated_fdog/255, img)

			plt.bone()
			plt.clf()
			plt.axis('off')
			plt.figimage(updated_fdog)
			dpi = 100
			print(updated_fdog.shape)
			plt.gcf().set_size_inches((updated_fdog.shape[1]/float(dpi),updated_fdog.shape[0]/float(dpi)))
			plt.savefig("fdog_iter"+str(iter_no)+".png",dpi=dpi) 
		# cv2.imshow('image',updated_fdog)
		# # cv2.imshow('abstract_image',abstract_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		

		self.fdog = updated_fdog

	def FlowBilateralFilter(self):
		def getNextPointInDirection(point, direction):
			"""
			point is a tuple
			direction is a tuple with magnitude 1
			"""
			x, y = point
			# print(point)
			dx, dy = direction
			# print(direction)
			nx, ny = x+dx, y+dy
			# print(nx, ny)
			return np.array([round(nx), round(ny)])

		def getPointOnLine(point): # 1x1, 3x3, 5x5, 7x7 ....
			x, y = point
			dx, dy = self.etf[x][y]
			dx, dy = -dy, dx
			px, py = x+2*dx, y+2*dy
			nx, ny = x-2*dx, y-2*dy

			ppoints = [np.array(p) for p in bresenham(int(x), int(y), int(px), int(py))]
			npoints = [np.array(p) for p in bresenham(int(x), int(y), int(nx), int(ny))]

			points = list(reversed(list(npoints)[1:self.config.fbl_T//2])) + list(ppoints[:self.config.fbl_T//2])
			points = [p for p in points if 0 <= p[0] < self.M and 0 <= p[1] < self.N]

			return points

		def getPointOnCurve(point): # 1x1, 3x3, 5x5, 7x7 ....
			# The points are not going to be in any order whatsoever
			# Use the array as a unordered set of points on the curve
			points = []
			for m in range(-1, 2, 2):
				npoint = point
				for s in range(self.config.fbl_S//2):
					x, y = npoint
					dx, dy = self.etf[int(x)][int(y)]
					npoint = getNextPointInDirection((x,y), (m*dx, m*dy))
					if 0 <= npoint[0] < self.M and 0 <= npoint[1] < self.N:
						points.append(npoint)
					else:
						break

			points.append(point)
			return points

		def spatialDomainGaussian(point, points, gaussian):
			# This implementation departs from what is done in FDoG
			# !!!!IMPORTANT!!!!
			# gaussian should be applied on the distance rather than s
			
			distances = np.linalg.norm(points-point, axis=1)

			return gaussian.pdf(distances)

		def colorDomainGaussian(point, points, gaussian):

			return gaussian.pdf(points-point)

		def filterAlongCurve(image, space_gaussian, color_gaussian):
			filter_image = np.zeros(image.shape)
			for i in range(self.M):
				for j in range(self.N):
					point = np.array([i,j])
					points = np.array(getPointOnCurve(point), dtype=np.int64)
					space_weights = np.array(spatialDomainGaussian(np.array((i,j)), points, space_gaussian))
					color_weights = np.array(colorDomainGaussian(image[i][j], image[points[:,0], points[:, 1]], color_gaussian))
					intensities = image[points[:, 0], points[:, 1]]

					filter_image[i][j] = np.einsum('ij,i->j',intensities, (space_weights * color_weights)) / np.sum(space_weights * color_weights)

			return filter_image

		def filterAlongGradient(image, space_gaussian, color_gaussian):
			filter_image = np.zeros(image.shape)
			for i in range(self.M):
				for j in range(self.N):
					point = np.array([i,j])
					points = np.array(getPointOnLine(point), dtype=np.int64)
					space_weights = np.array(spatialDomainGaussian(np.array((i,j)), points, space_gaussian))
					color_weights = np.array(colorDomainGaussian(image[i][j], image[points[:,0], points[:, 1]], color_gaussian))
					intensities = image[points[:, 0], points[:, 1]]

					filter_image[i][j] = np.einsum('ij,i->j',intensities, (space_weights * color_weights)) / np.sum(space_weights * color_weights)

			return filter_image

		e_space_gaussian = scipy.stats.norm(0, self.config.fbl_sigma_e)
		g_space_gaussian = scipy.stats.norm(0, self.config.fbl_sigma_g)
		e_color_gaussian = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=np.diag([self.config.fbl_r_e]*3))
		g_color_gaussian = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=np.diag([self.config.fbl_r_g]*3))

		filter_image = np.copy(self.image)
		for ite in range(self.config.fbl_iter):
			print("Along curve underway")
			filter_image = filterAlongCurve(filter_image, e_space_gaussian, e_color_gaussian)
			plt.bone()
			plt.clf()
			plt.axis('off')
			plt.figimage(filter_image)
			dpi = 100
			plt.gcf().set_size_inches((filter_image.shape[1]/float(dpi),filter_image.shape[0]/float(dpi)))
			plt.savefig("fbl_eiter"+str(ite)+".png",dpi=dpi) 
			print("Along grad underway")
			filter_image = filterAlongGradient(filter_image, g_space_gaussian, g_color_gaussian)
			plt.bone()
			plt.clf()
			plt.axis('off')
			plt.figimage(filter_image)
			dpi = 100
			plt.gcf().set_size_inches((filter_image.shape[1]/float(dpi),filter_image.shape[0]/float(dpi)))
			plt.savefig("fbl_giter"+str(ite)+".png",dpi=dpi) 

		self.smoothing = filter_image

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
		KSby2 = self.config.etf_radius
		H, W, C = self.image.shape

		indices = np.indices([H, W]).transpose((1,2,0))

		ite = 0
		util.view_vf(vector_field,ite)
		ite += 1

		updated_vf = np.zeros(vector_field.shape)
		for iter_no in range(self.config.etf_iterations):
			"""
			padded_mag = cv2.copyMakeBorder()
			"""
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
		if os.path.isfile('etf_elvis.npy'):
			self.etf = np.load('etf.npy')
		else:
			self.ETF()
			np.save("etf_elvis.npy", self.etf)
		# self.FDOG()
		# self.etf = np.zeros([self.M, self.N, 2])
		self.FlowBilateralFilter()
		return self.smoothing


