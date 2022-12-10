import os
import math

import cv2
import numpy as np
import pylab as plt
import scipy.stats
# from bresenham import bresenham

import src.util as util


class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config
		self.M, self.N, self.C = image.shape

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
				points[i][0] = math.floor(y+0.5)
				points[i][1] = x
				points[size-i][0] = -points[i][0]
				points[size-i][1] = -points[i][1]
				i +=1
		else:
			slope_ = 1/slope
			for y in range(size//2):
				x += slope_
				y += 1
				points[i][0] = y
				points[i][1] = math.floor(x+0.5)
				points[size-i][0] = -points[i][0]
				points[size-i][1] = -points[i][1]
				i +=1

		return points

	def getNextPointOnCurve(self, slope): #1X1 3X3 5X5 7X7
		point = np.zeros((2))
		if abs(slope)<=1:
			point[0] = math.floor(slope+0.5)
			point[1] = 1
		else:
			slope_ = 1/slope
			point[1] = math.floor(slope_+0.5)
			point[0] = 1
		return point

	def getF(self): #1X1 3X3 5X5 7X7
		sigma_c = self.config.sigma_c
		p = self.config.rho
		points = []
		t = 0
		distrib_c = scipy.stats.norm(0, sigma_c)
		distrib_s = scipy.stats.norm(0, 1.6*sigma_c)
		points.append([distrib_c.pdf(t),distrib_s.pdf(t)])
		t += 1
		# while t<5:
		while True:
			c = distrib_c.pdf(t)
			s = distrib_s.pdf(t)
			if s < 0.001:
				break
			points.append([c, s])
			t += 1
		size = len(points)
		i = size-1
		while i != 0:
			points.append(points[i])
			i-=1
		np_points = np.array(points)
		sum_gauss = np.sum(np_points,axis=0)
		p = sum_gauss[1]/sum_gauss[0]
		points = [pt[0]-p*pt[1] for pt in points]
		print(p)
		print(points)
		print(np.sum(np.array(points)))
		return points

	def getG(self): #1X1 3X3 5X5 7X7
		sigma_m = self.config.sigma_m
		points = []
		t = 0
		distrib = scipy.stats.norm(0, sigma_m)
		points.append(distrib.pdf(t))
		t += 1
		# while t<5:
		while True:
			m = distrib.pdf(t)
			if m < 0.001:
				break
			points.append(m)
			t += 1
		size = len(points)
		i = size-1
		while i != 0:
			points.append(points[i])
			i-=1
		return points

	def FDOG(self):
		img = np.float64(self.preProcess)/255
		fdog = np.copy(img)
		updated_fdog = np.zeros(fdog.shape)
		M = self.image.shape[0]
		N = self.image.shape[1]
		f = np.array(self.getF())
		f_kernel_size = f.shape[0]
		print(f_kernel_size)
		print(f)
		g = np.array(self.getG())
		g_kernel_size = g.shape[0]
		print(g_kernel_size)
		print(g)

		for iter_no in range(self.config.fdog_iterations):
			Hg = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					t0 = self.etf[i][j]
					direc = math.atan2(t0[0],-t0[1]);
					points = self.getPointsOnLine(direc, f_kernel_size).astype(np.int)
					# print(str(direc)+" "+str(points))
					ty = [1 if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					I = [fdog[i+p[0]][j+p[1]] if i+p[0]<M and i+p[0]>=0 and j+p[1]<N and j+p[1]>=0 else 0 for p in points]
					Hg[i][j] = np.sum(I*f)/np.sum(f*ty)
					# if(Hg[i][j]) < 0:
						# print("YES")
			# print(np.sum(np.array(Hg) < 0))
			He = np.zeros(fdog.shape)
			for i in range(M):
				for j in range(N):
					point = np.array([i,j])
					H = Hg[point[0]][point[1]]*g[0]
					G = g[0]
					for s in range(1,g_kernel_size//2+1):
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
					for s in range(1,g_kernel_size//2+1):
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
					# print(str(val)+" " + str(1+np.tanh(val)))
					if val<0:
						He[i][j] = 1+np.tanh(val)
					else:
						He[i][j] = 1
			# He = cv2.normalize(He, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

			print("num 0 in He" + str(np.sum(np.array(He) > 0)))
			# updated_fdog = He
			np.save("result/fdog1_"+self.config.image_path+".npy", He)
			for i in range(M):
				for j in range(N):
					updated_fdog[i][j] = 0 if He[i][j] < self.config.fdog_threshold else 255
					if updated_fdog[i][j]==0:
						fdog[i][j] = 0
			fdog = cv2.GaussianBlur(fdog, (3,3), 0, 0, cv2.BORDER_DEFAULT)
			util.save_image(updated_fdog,self.config.image_path,"fdog_iter_"+str(iter_no+1))
			util.save_image(fdog,self.config.image_path,"superimpose_iter_"+str(iter_no+1))
		self.fdog = updated_fdog

	def FlowBilateralFilter(self):
		# def getNextPointInDirection(point, direction):
		# 	"""
		# 	point is a tuple
		# 	direction is a tuple with magnitude 1
		# 	"""
		# 	x, y = point
		# 	# print(point)
		# 	dx, dy = direction
		# 	# print(direction)
		# 	nx, ny = x+dx, y+dy
		# 	# print(nx, ny)
		# 	return np.array([round(nx), round(ny)])

		# def getPointOnLine(point): # 1x1, 3x3, 5x5, 7x7 ....
		# 	x, y = point
		# 	dx, dy = self.etf[x][y]
		# 	dx, dy = -dy, dx
		# 	px, py = x+2*dx, y+2*dy
		# 	nx, ny = x-2*dx, y-2*dy

		# 	ppoints = [np.array(p) for p in bresenham(int(x), int(y), int(px), int(py))]
		# 	npoints = [np.array(p) for p in bresenham(int(x), int(y), int(nx), int(ny))]

		# 	points = list(reversed(list(npoints)[1:self.config.fbl_T//2])) + list(ppoints[:self.config.fbl_T//2])
		# 	points = [p for p in points if 0 <= p[0] < self.M and 0 <= p[1] < self.N]

		# 	return points

		# def getPointOnCurve(point): # 1x1, 3x3, 5x5, 7x7 ....
		# 	# The points are not going to be in any order whatsoever
		# 	# Use the array as a unordered set of points on the curve
		# 	points = []
		# 	for m in range(-1, 2, 2):
		# 		npoint = point
		# 		for s in range(self.config.fbl_S//2):
		# 			x, y = npoint
		# 			dx, dy = self.etf[int(x)][int(y)]
		# 			npoint = getNextPointInDirection((x,y), (m*dx, m*dy))
		# 			if 0 <= npoint[0] < self.M and 0 <= npoint[1] < self.N:
		# 				points.append(npoint)
		# 			else:
		# 				break

		# 	points.append(point)
		# 	return points

		def spatialDomainGaussian(space_gaussian):
			# This implementation departs from what is done in FDoG
			# !!!!IMPORTANT!!!!
			# gaussian should be applied on the distance rather than s
			
			points = []
			t = 0
			points.append(space_gaussian.pdf(t))
			t += 1
			# while t<5:
			while True:
				m = space_gaussian.pdf(t)
				if m < 0.001:
					break
				points.append(m)
				t += 1
			size = len(points)
			i = size-1
			while i != 0:
				points.append(points[i])
				i-=1
			return points

		def colorDomainGaussian(point, points, gaussian):

			return gaussian.pdf(points-point)

		def filterAlongCurve(image, space_gaussian, color_gaussian):
			filter_image = np.zeros(image.shape)
			g = np.array(spatialDomainGaussian(space_gaussian))
			g_kernel_size = g.shape[0]
			print(g_kernel_size)
			print(g)
			for i in range(self.M):
				for j in range(self.N):
					point = np.array([i,j])
					H = image[point[0]][point[1]]*g[0]
					G = g[0]
					for s in range(1,g_kernel_size//2+1):
						s_etf = self.etf[point[0]][point[1]]
						direc = math.atan2(s_etf[1],s_etf[0])
						p_delta = self.getNextPointOnCurve(direc).astype(np.int)
						point = point + p_delta
						if point[0]<self.M and point[0]>=0 and point[1]<self.N and point[1]>=0:
							color_weights = color_gaussian.pdf(image[i][j]-image[point[0]][point[1]])
							H += image[point[0]][point[1]]*g[s]*color_weights
							G += g[s]*color_weights
						else:
							break
					point = np.array([i,j])
					for s in range(1,g_kernel_size//2+1):
						s_etf = self.etf[point[0]][point[1]]
						direc = math.atan2(-s_etf[1],-s_etf[0])
						p_delta = self.getNextPointOnCurve(direc).astype(np.int)
						point = point + p_delta
						if point[0]<self.M and point[0]>=0 and point[1]<self.N and point[1]>=0:
							color_weights = color_gaussian.pdf(image[i][j]-image[point[0]][point[1]])
							H += image[point[0]][point[1]]*g[s]*color_weights
							G += g[s]*color_weights
						else:
							break
					filter_image[i][j] = H/G
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

		def colorQuantization(image):
			config = self.config
			lab_space = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
			gray_img = lab_space[:,:,0]
			sobelx = cv2.Sobel(gray_img,cv2.CV_32F,1,0,ksize=3)
			sobely = cv2.Sobel(gray_img,cv2.CV_32F,0,1,ksize=3)

			mag, angle = cv2.cartToPolar(sobelx, sobely, magnitude=None, angle=None, angleInDegrees=True)
			mag = np.clip(mag, a_min=config.fbl_cq_gradient_min, a_max=config.fbl_cq_gradient_max)
			phi_q = np.interp(mag, [config.fbl_cq_gradient_min,config.fbl_cq_gradient_max],[config.fbl_cq_target_sharp_min,config.fbl_cq_target_sharp_max])

			x = gray_img.astype(np.uint8)
			q_nearest = x - x%config.fbl_cq_q + np.around(2*(x%config.fbl_cq_q)/config.fbl_cq_q)*(config.fbl_cq_q/2)
			
			final_image = q_nearest + (config.fbl_cq_q/2)*np.tanh(np.multiply(phi_q, gray_img-q_nearest))
			lab_space[:,:,0] = final_image.astype(np.uint8)

			return cv2.cvtColor(lab_space, cv2.COLOR_LAB2BGR)

		e_space_gaussian = scipy.stats.norm(0, self.config.fbl_sigma_e)
		g_space_gaussian = scipy.stats.norm(0, self.config.fbl_sigma_g)
		e_color_gaussian = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=np.diag([self.config.fbl_r_e]*3))
		g_color_gaussian = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=np.diag([self.config.fbl_r_g]*3))

		filter_image = np.copy(self.image)
		for ite in range(self.config.fbl_iter):
			print("Along curve underway")
			filter_image = filterAlongCurve(filter_image.astype(np.float64), e_space_gaussian, e_color_gaussian).astype(np.uint8)
			display_img = cv2.cvtColor(filter_image, cv2.COLOR_BGR2RGB)
			util.save_image(display_img,self.config.image_path,"fbl_eiter"+str(ite))
			# print(np.sum(np.linalg.norm(filter_image-self.image.astype(np.float64))))
			print("Along grad underway")
			filter_image = filterAlongGradient(filter_image.astype(np.float64), g_space_gaussian, g_color_gaussian).astype(np.uint8)
			display_img = cv2.cvtColor(filter_image, cv2.COLOR_BGR2RGB)
			util.save_image(display_img,self.config.image_path,"fbl_giter"+str(ite))

		# image = cv2.imread('results/fbl_giter4.png')
		self.smoothing = colorQuantization(filter_image)
		display_img = cv2.cvtColor(self.smoothing, cv2.COLOR_BGR2RGB)
		util.save_image(display_img,self.config.image_path,"fbl_quantize")

	def preProcess(self):
		smoothen_image = cv2.GaussianBlur(self.image, (5,5), 0, 0, cv2.BORDER_DEFAULT)
		gray_img = cv2.cvtColor(smoothen_image, cv2.COLOR_BGR2GRAY)
		self.preProcess = gray_img

	def ETF(self):
		gray_img = self.preProcess
		sobelx = cv2.Sobel(gray_img,cv2.CV_32F,1,0,ksize=3)
		sobely = cv2.Sobel(gray_img,cv2.CV_32F,0,1,ksize=3)

		mag, angle = cv2.cartToPolar(sobelx, sobely, magnitude=None, angle=None, angleInDegrees=True)
		mag = cv2.normalize(mag, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
		angle = (angle + 90) % 360
		x, y = cv2.polarToCart(mag, angle, x=None, y=None, angleInDegrees=True)
		vector_field = np.stack([x, y], axis=2)

		KSby2 = self.config.etf_radius
		H, W, C = self.image.shape

		indices = np.indices([H, W]).transpose((1,2,0))

		ite = 0
		util.view_vf(vector_field,ite,self.config.image_path)
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
			util.view_vf(updated_vf,ite,self.config.image_path)
			vector_field = updated_vf
			ite += 1

		self.etf = updated_vf

	def thresholding(self,value):
		He = np.load("result/fdog1_"+self.config.image_path+".npy")[:,:,0]
		updated_fdog = np.zeros(He.shape)
		print(He)
		for i in range(He.shape[0]):
			for j in range(He.shape[1]):
				updated_fdog[i][j] = 0 if He[i][j] < value else 255
		updated_fdog = 255 - updated_fdog
		self.fdog = updated_fdog
		# updated_fdog = cv2.GaussianBlur(updated_fdog, (3,3), 0, 0, cv2.BORDER_DEFAULT)
		util.save_image(updated_fdog,self.config.image_path,"fdog_iter_"+str(0+1))

	def combine_image(self):
		# self.output = np.zeros(self.image.shape)
		temp = np.where(self.fdog==0,0,1)
		print(np.sum(temp))
		print(np.sum(self.fdog))
		print(self.fdog)
		self.output = np.einsum("ij,ijk->ijk",temp,self.smoothing).astype(np.uint8)
		self.output = cv2.cvtColor(self.output, cv2.COLOR_BGR2RGB)
		# for i in range(self.image.shape[0]):
		# 	for j in range(self.image.shape[1]):
		# 		if self.fdog[i][j]=0:
		# 			self.output[i][j] = 0
		# 		else:
		# 			self.output[i][j] = self.smoothing[i][j]
		util.save_image(self.output,self.config.image_path,"final")


	def run(self):
		self.preProcess()
		if os.path.isfile('result/etf_'+self.config.image_path+'.npy'):
			self.etf = np.load('result/etf_'+self.config.image_path+'.npy')
		else:
			self.ETF()
			np.save("result/etf_"+self.config.image_path+".npy", self.etf)
		if os.path.isfile('result/fdog1_'+self.config.image_path+'.npy'):
			self.fdog = np.load('result/fdog1_'+self.config.image_path+'.npy')
		else:
			self.FDOG()
			np.save("result/fdog1_"+self.config.image_path+".npy", self.etf)
		# self.FDOG()
		# self.thresholding(0.5)
		self.FlowBilateralFilter()
		self.combine_image()
		return self.output


