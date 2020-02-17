# importing libraries 
import numpy as np 
import imutils
import cv2 

field_threshold = {"prev_policy_no": 0.9,
                    "address": 0.73,
                    "tamil": 0.8,
                    "Vivek_Gupta": 0.59,
                    "Rakesh": 0.8
                    } 


# Function to Generate bounding 
# boxes around detected fields 
def getBoxed(img, img_gray, template, field_name="policy_no"):
	w, h = template.shape[::-1]

	# Apply template matching 
	res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) 

	hits = np.where(res >= field_threshold[field_name]) 

	# Draw a rectangle around the matched region. 
	for pt in zip(*hits[::-1]): 
		cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2) 

		y = pt[1] - 10 if pt[1] - 10 > 10 else pt[1] + h + 20

		cv2.putText(img, field_name, (pt[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1) 

	return img 


# Driver Function 
if __name__ == '__main__': 

	# Read the original document image 
	# img = cv2.imread('doc.png')
	# img_me = cv2.imread('me2.jpg')
	img = cv2.imread('me2.jpg')
	# cv2.imshow("Mr Vivek Gupta",img)
		
	# 3-d to 2-d conversion 
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	
	# Field templates 
	# template_add = cv2.imread('doc_address.png', 0) 
	# template_prev = cv2.imread('doc_prev_policy.png', 0)
	# template_tamil = cv2.imread('tamil.PNG', 0)
	template_me = cv2.imread('me.png', 0)
	template_rakesh = cv2.imread('rakesh.png', 0)

	# img = getBoxed(img.copy(), img_gray.copy(), template_add, 'address') 
	# img = getBoxed(img.copy(), img_gray.copy(), template_prev, 'prev_policy_no')
	# img = getBoxed(img.copy(), img_gray.copy(), template_tamil, 'tamil')
	img = getBoxed(img.copy(), img_gray.copy(), template_me, 'Vivek_Gupta')
	img = getBoxed(img.copy(), img_gray.copy(), template_rakesh, 'Rakesh')

	# cv2.imwrite("Document_Output.png", img)
	cv2.imwrite("Me_Rakesh_Output.jpg", img)
	cv2.imshow('Detected', img) 
