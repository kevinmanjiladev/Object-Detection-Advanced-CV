import cv2
import numpy as np
from fastapi import FastAPI,UploadFile,File,Request
import mysql.connector
import datetime
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app=FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# load cascade
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load trained model
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")


# Label map
label_map={
    0:"Kevin",
    1:'Martin',
    2:'Nazim',
    3:'Yadukrishnan'
}

# marked=set()
# import mysql.connector

# connect to mysql
conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="pass123",
    database="attendence_db"
)
cursor=conn.cursor()

def marked_attendence(name):
    import datetime
    now=datetime.datetime.now()
    date=now.strftime("%Y-%m-%d")
    time=now.strftime("%H:%M:%S")

    check_query="SELECT * FROM attendence WHERE name=%s and date=%s;"
    cursor.execute(check_query,(name,date))
    result=cursor.fetchone()
    if result is None:
        insert_query="INSERT INTO attendence (name,date,time) VALUES(%s,%s,%s)"
        cursor.execute(insert_query,(name,date,time))
        status="Marked"
        conn.commit()
    else:
        status ="Already Marked"
    return status


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="recognize.html",
        )


@app.post('/recognize')
async def recognize(file:UploadFile=File(...)):
    image_bytes=await file.read()
    np_arr=np.frombuffer(image_bytes,np.uint8)
    frame=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    results=[]

    for (x,y,w,h) in faces:
        face_roi=gray[y:y+h,x:x+w]
        face_roi=cv2.resize(face_roi,(200,200))
        label,confidence=recognizer.predict(face_roi)
        if confidence<100:
            name=label_map.get(label,"Unknown")
            status=marked_attendence(name)
            results.append({
                "name":name,
                "confidence":float(confidence),
                "status":status
            })
        else:
            results.append({
                "name":"Unknown",
                "confidence":float(confidence),
                "status":"Not Recognized"
            })
    return {"results":results}  