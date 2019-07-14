from imutils import face_utils
import dlib
import cv2
import pyautogui as pag
import time
from pynput.mouse import Button, Controller
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

# dimensions of the screen are 1920x1080.
def dst(point1,point2):
	distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance 

def EAR(point1,point2,point3,point4,point5,point6):
	ear = (dst(point2,point6) + dst(point3,point5))/(2*dst(point1,point4))*1.0
	return ear
	
def angle(point1):
	# In OpenCV, since the x and y coordinates start from the top right corner of the image or the image is in the fourth quadrant, the conventions applied are in reverse.
	slope12 = (point1[1] - 250)/(point1[0] - 250)*1.0
	angle = -1.0*np.arctan(slope12)
	return angle

def MAR(point1,point2,point3,point4,point5,point6,point7,point8):
	mar=(dst(point2,point8) + dst(point3,point7) + dst(point4,point6))/(2*dst(point1,point5))*1.0
	return mar

def AREA(point1,point2,point3,point4,point5,point6):
	area=(abs((point1[0]*point2[1])+(point2[0]*point3[1])+(point3[0]*point4[1])+(point4[0]*point5[1])+(point5[0]*point6[1])+(point6[0]*point1[1])-(point2[0]*point1[1])-\
(point3[0]*point2[1])-(point4[0]*point3[1])-(point5[0]*point4[1])-(point6[0]*point5[1])-(point1[0]*point6[1])))/2
	return area

#mouse = Controller()
# Right eye: 37 to 42, Left Eye: 43 to 48
p = "shape_predictor.dat"
detector = dlib.get_frontal_face_detector() # Returns a default face detector object
predictor = dlib.shape_predictor(p) # Outputs a set of location points that define a pose of the object. (Here, pose of the human face)
cap = cv2.VideoCapture(0)
initial=time.time()
final1=initial+1
final2=final1+5
final3=final2+5
final4=final3+1
final5=final4+5
final6=final5+1
final7=final6+5
final8=final7+2
final9=final8+5
final10=final9+1
final11=final10+5
opened=np.array([])
left=np.array([])
right=np.array([])
mouth=np.array([])
areal=np.array([])
arear=np.array([])
t0=np.array([])
t1=np.array([])
t2=np.array([])
t3=np.array([])
t4=np.array([])
t5=np.array([])


while ((time.time()-initial)<=38):
	    # Getting out image by webcam 
	_, image = cap.read() 
	image=cv2.flip(image,1)
	    # Converting the image to gray scale image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)	
	    # Get faces into webcam's image
	rects = detector(gray, 0)
	#if (len(rects))>1:
	#	rects=rects[0]
	#elif (len(rects))==1:
	#	rects=rects
	
	    # For each detected face, find the landmark.
	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

		actual_t=time.time()-initial
		l=EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		r=EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		l_a=AREA(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		r_a=AREA(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		sub_area=(l_a - r_a)
		sub=(l-r)*100

		if (final1<time.time()) and (time.time()<final2):
			cv2.putText(image,'callibration is going to start....',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			opened=np.append(opened,[sub])
			t0=np.append(t0,[actual_t])
		
		elif (final2<time.time()) and (time.time()<final3):
			cv2.putText(image,'blink left eye',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			#l=EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
			left=np.append(left,[sub])
			t1=np.append(t1,[actual_t])
										
		elif (final4<time.time()) and (time.time()<final5):
			cv2.putText(image,'blink right eye',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			#r=EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
			right=np.append(right,[sub])
			t2=np.append(t2,[actual_t])

		elif (final6<time.time()) and (time.time()<final7):
			cv2.putText(image,'open mouth',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			m=MAR(shape[60],shape[61],shape[62],shape[63],shape[64],shape[65],shape[66],shape[67])
			mouth=np.append(mouth,[m])
			t3=np.append(t3,[actual_t])
	
		elif (final8<time.time()) and (time.time()<final9):
			cv2.putText(image,'blink left eye',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			areal=np.append(areal,[sub_area])
			t4=np.append(t4,[actual_t])

		elif (final10<time.time()) and (time.time()<final11):
			cv2.putText(image,'blink right eye',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
			arear=np.append(arear,[sub_area])
			t5=np.append(t5,[actual_t])
	
		else:
			pass

	cv2.imshow('frame',image)
	if cv2.waitKey(5) & 0xFF==27:
		break

plt.subplot(3,2,1)
smooth_1=gaussian_filter(left,sigma=5)
plt.title('fig1')
plt.plot(t1,smooth_1)
plt.subplot(3,2,2)
smooth_2=gaussian_filter(right,sigma=5)
plt.title('fig2')
plt.plot(t2,smooth_2)
plt.subplot(3,2,3)
smooth_3=gaussian_filter(mouth,sigma=5)
plt.title('fig3')
plt.plot(t3,smooth_3)
plt.subplot(3,2,4)
smooth_0=gaussian_filter(opened,sigma=5)
plt.title('fig0')
plt.plot(t0,smooth_0)
plt.subplot(3,2,5)
smooth_4=gaussian_filter(areal,sigma=5)
plt.title('fig4')
plt.plot(t4,smooth_4)
plt.subplot(3,2,6)
smooth_5=gaussian_filter(arear,sigma=5)
plt.title('fig5')
plt.plot(t5,smooth_5)


plt.show()
cap.release()
cv2.destroyAllWindows()

cap=cv2.VideoCapture(0)
meanm=np.mean(mouth)
meanl=np.mean(left)
meanr=np.mean(right)
meano=np.mean(opened)
meanareal=np.mean(areal)
meanarear=np.mean(arear)
MARlist=np.array([])
scroll_status=0
while True:
	try:
		_,image=cap.read()
		image=cv2.flip(image,1)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		gray=cv2.equalizeHist(gray)
		rects=detector(gray,0)
		for (i,rect) in enumerate(rects):
			shape=predictor(gray,rect)
			shape=face_utils.shape_to_np(shape)
			[h,k]=shape[33]
			cv2.circle(image,(h,k),1,(255,0,0),2)
			cv2.line(image,(250,270),(h,k),(0,0,0),1)
			cv2.circle(image,(250,270),50,(0,0,255),2)
			mar = MAR(shape[60],shape[61],shape[62],shape[63],shape[64],shape[65],shape[66],shape[67]) 
			MARlist = np.append(MARlist,[mar]) # Appending the list at every iteration
			if len(MARlist) == 30: # till it reaches a size of 30 elements
				mar_avg = np.mean(MARlist)
				MARlist = np.array([]) # Resetting the MAR list
				if (int(mar_avg*100) > int(meanm*100)):
					if scroll_status == 0:
						scroll_status = 1
					else:
						scroll_status = 0
			if scroll_status == 0:
				print('scroll mode is off')
				if((h-250)**2 + (k-270)**2 - 50**2 > 0):
					a = angle(shape[33]) # Calculates the angle
					if h > 250: # The below conditions set the conditions for the mouse to move and that too in any direction we desire it to move to.
						pag.moveTo(pag.position()[0]+(10*np.cos(-1.0*a)),pag.position()[1]+(10*np.sin(-1.0*a)),duration = 0.01)
					else:
						pag.moveTo(pag.position()[0]-(10*np.cos(-1.0*a)),pag.position()[1]-(10*np.sin(-1.0*a)),duration = 0.01)
			else: #Enabling scroll status
				cv2.putText(image,'Scroll mode ON',(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
				print('scroll mode is on')
				if k > 320: 
					pag.scroll(-1)
				elif k < 220:
					pag.scroll(1)
				else:
					pass
		for (x,y) in shape:
			cv2.circle(image,(x,y),1,(0,255,0),1)
			
		
		#right_eye=EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		#left_eye=EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		left_area=AREA(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		right_area=AREA(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		#mar = MAR(shape[60],shape[61],shape[62],shape[63],shape[64],shape[65],shape[66],shape[67]) 
		#real_diff=(left_eye-right_eye)*100
		area_diff=(left_area - right_area)
		if  (area_diff < (meanareal-22 )) :
			pag.click(button='left')
			print('left blink')
		elif (area_diff > (meanarear+25)):
			pag.click(button='right')
			print('right blink')
		else:
			pass

		
		
		cv2.imshow('frame',image)
		if (cv2.waitKey(5) & 0xFF) == ord('h'):
			break
	
	except ValueError:
		print('Paused')
		if (cv2.waitKey(5) & 0xFF) == ord('h'):
			break
	
cap.release()
cv2.destroyAllWindows()

	




