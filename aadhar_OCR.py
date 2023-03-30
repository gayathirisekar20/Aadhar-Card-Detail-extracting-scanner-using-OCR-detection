import cv2
import numpy as np
import pytesseract
import re
from pytesseract import Output


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
cap = cv2.VideoCapture('./images/4.jpg')
  
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret, img = cap.read()
  if ret == True:
    # Display the resulting frame
    # ret, img = cv2.threshold(img, 80, 255 , cv2.THRESH_BINARY)
    cv2.imshow('Frame', img)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break
   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()
img=cv2.imread("./images/7.jpg")
rgb_planes = cv2.split(img)
result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))        #change the value of (10,10) to see different results
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                    dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)
dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)             # removing noise from image
    
text = pytesseract.image_to_string((dst),lang ="eng",config=tessdata_dir_config )
# print(text)
# text = pytesseract.image_to_string(dst).upper().replace(" ", "")

date = str(re.findall(r"[\d]{1,2}[/-][\d]{1,2}[/-][\d]{1,4}", text)).replace("]", "").replace("[","").replace("'", "")
print(date)
# number = str(re.findall(r"[0-9]{11,12}", text)).replace("]", "").replace("[","").replace("'", "")

# print(num)
num=text.replace(date, '')
# num= text.strip(b)
# print (num)
num=str(re.findall(r'([0-9]{4})+',num)).replace("[","").replace("'","").replace("]","")
num=(re.sub('[,+]','',num))
print(num)


sex = str(re.findall(r"MALE|FEMALE", text)).replace("[","").replace("'", "").replace("]", "")
print(sex)

text=text.replace("Government Of India ", '')
name = str(re.findall(r"([A-Z][a-z]+)", text)).replace("[","").replace("'", "").replace("]", "")
name=(re.sub('[,+]','',name))
print(name)
# cv2.imshow('original',img)
# cv2.imshow('edited',dst)
    

# crop_pic from ID card

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 7)
for (x, y, w, h) in faces:
    ix = 0
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = img[y:y + h, x:x + w]
    #crop_pic = cv2.imwrite('croppic10.jpg', roi_color)
    # crop_pic = cv2.imshow('croppicds', roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
