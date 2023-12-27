import cv2

fc=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
ec=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
sc=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

def detector(img,frame):
    x=0
    face=fc.detectMultiScale(img,1.3,10)
    for(x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),((x+w),(y+h)),(255,0,0),2)
        img2=img[y:y+h,x:x+w]
        color=frame[y:y+h,x:x+w]
        smile=sc.detectMultiScale(img2,1.05,20)
        for(sx,sy,sw,sh) in smile:
                x=1
                cv2.rectangle(color,(sx,sy),((sx+sw),(sy+sh)),(0,0,255),2)
        if x==1:
            cv2.putText(color,'NOT WEARING MASK',(55,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,cv2.LINE_AA)
        else:
            cv2.putText(color,'WEARING MASK',(55,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1,cv2.LINE_AA)

    return frame

webcam=cv2.VideoCapture(0)
webcam.open(0)
while webcam.isOpened():
    _,frame=webcam.read()
    img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    can=detector(img,frame)
    cv2.imshow('mask check',can)
    if cv2.waitKey(1) & 0xff==27:
        break
webcam.release()
cv2.destroyAllWindows()
