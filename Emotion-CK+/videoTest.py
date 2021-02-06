'''
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
'''
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'models/model.h5'
#model=load_model('./models/model.h5')

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)#, compile=False)
EMOTIONS = ("disgust" ,"anger","fear", "surprise", "happy", "contempt", "sadness")

face_haar_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# model_shape=model.input_shape[1:3]
#cv2.namedWindow('Emotion')
camera=cv2.VideoCapture(0)
#cap=cv2.VideoCapture(0)
while True:
	ret,frame = camera.read()
	if not ret:
		continue

	frame=imutils.resize(frame,width=300)
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5)#,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	canvas = np.zeros((250,300,3), dtype="uint8")
	frameClone=frame.copy()
	for (fX,fY,fW,fH) in faces:
#		cv2.rectangle(frame,(fX,fY),(fX+fW,fY+fH),(255,0,0),thickness=2)
#		gray=frame[fY:fY+fW,fX:fX+fH]

#		canvas = np.zeros((250,300,3), dtype="uint8")
#		frameClone=frame.copy()
	#if len(faces)>0:
	#	faces=sorted(faces,reverse=True,key=lambda x:(x[2]-x[0])*(x[3]-x[1]))[0]
	#	(fX,fY,fW,fH)=faces
		roi=frame[fY:fY+fH,fX:fX+fW]
		roi=cv2.resize(roi,(48,48))
#		roi=roi.astype("float")/255.0
		roi=img_to_array(roi)
		roi=np.expand_dims(roi,axis=0)
		roi/=255

#		preds=emotion_classifier.predict(roi)[0]
#		emotion_probability=np.argmax(preds)
#		label=EMOTIONS[emotion_probability]
		preds=emotion_classifier.predict(roi)
		emotion_probability=np.argmax(preds[0])
		label=EMOTIONS[emotion_probability]

		#max_index=np.argmax(preds[0])
		#EMOTIONS = ("disgust" ,"anger","fear", "surprise", "happy", "contempt", "sadness")
		#label=EMOTIONS[max_index]
	
#	else:
#		continue

	for (i,(emotion,prob)) in enumerate(zip(EMOTIONS,preds[0])):
		text="{}:{:.2f}%".format(emotion,prob*100)

		w = int(prob*300)
		cv2.rectangle(canvas,(7,(i*35)+5),(w,(i*35)+35),(0,0,255),-1)
		cv2.putText(canvas,text,(10,(i*35)+23),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)
		cv2.putText(frameClone,label,(fX,fY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
		cv2.rectangle(frameClone,(fX,fY),(fX+fW,fY+fH),(0,0,255),2)
	
	cv2.imshow("your face",frameClone)
	cv2.imshow("Prob",canvas)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
camera.release()
cv2.destroyAllWindows()


'''
	canvas = np.zeros((250,300,3), dtype="uint8")
	frameClone=frame.copy()
	if len(faces)>0:
		faces=sorted(faces,reverse=True,key=lambda x:(x[2]-x[0])*(x[3]-x[1]))[0]
		(fX,fY,fW,fH)=faces
		roi=frame[fY:fY+fH,fX:fX+fW]
		roi=cv2.resize(roi,(48,48))
		roi=roi.astype("float")/255.0
		roi=img_to_array(roi)
		roi=np.expand_dims(roi,axis=0)

		preds=emotion_classifier.predict(roi)[0]
#		emotion_probability=np.argmax(preds)
#		label=EMOTIONS[emotion_probability]
		emotion_probability=np.max(preds)
		label=EMOTIONS[preds.argmax()]

		#max_index=np.argmax(preds[0])
		#EMOTIONS = ("disgust" ,"anger","fear", "surprise", "happy", "contempt", "sadness")
		#label=EMOTIONS[max_index]
	
	else:
		continue

	for (i,(emotion,prob)) in enumerate(zip(EMOTIONS,preds)):
		text="{}:{:.2f}%".format(emotion,prob*100)

		w = int(prob*300)
		cv2.rectangle(canvas,(7,(i*35)+5),(w,(i*35)+35),(0,0,255),-1)
		cv2.putText(canvas,text,(10,(i*35)+23),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)
		cv2.putText(frameClone,label,(fX,fY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
		cv2.rectangle(frameClone,(fX,fY),(fX+fW,fY+fH),(0,0,255),2)
	
	cv2.imshow("your face",frameClone)
	cv2.imshow("Prob",canvas)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
camera.release()
cv2.destroyAllWindows()'''