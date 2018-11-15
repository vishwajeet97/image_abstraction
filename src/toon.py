class Toon:
	def __init__(self, image, config):
		self.image = image
		self.config = config

	def ETF(image):
		gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sobelx = cv.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
		sobely = cv.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
		sobelx = np.uint8(np.absolute(sobelx))
		sobely = np.uint8(np.absolute(sobely))
		mag, ang = cv2.cartToPolar(sobelx, sobely)
		KERNEL_SIZE = 5
		


