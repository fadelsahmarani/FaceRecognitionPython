#!/usr/bin/env python
import pyodbc
import sys
import os
import numpy as np
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.detectors import FaceDetector
import face_recognition_system.operations as op
import cv2
from cv2 import __version__

def get_images(frame, faces_coord, shape):

    if shape == "rectangle":
        faces_img = op.cut_face_rectangle(frame, faces_coord)
        frame = op.draw_face_rectangle(frame, faces_coord)
    elif shape == "ellipse":
        faces_img = op.cut_face_rectangle(frame, faces_coord)
        frame = op.draw_face_rectangle(frame, faces_coord)
    faces_img = op.normalize_intensity(faces_img)
    faces_img = op.resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder, shape):
    person_name = raw_input('What is the name of the new person: ').lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        raw_input("I will now take 20 pictures. Press ENTER when ready.")
        os.mkdir(folder)
        video = VideoCamera()
        detector = FaceDetector('face_recognition_system/frontal_face.xml')
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        while counter < 21:
            frame = video.get_frame()
            face_coord = detector.detect(frame)
            if len(face_coord):
                frame, face_img = get_images(frame, face_coord, shape)
                if timer % 100 == 5:
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                                face_img[0])

                    print('Images Saved:' + str(counter))
                    
                    counter += 1
                    cv2.imshow('Saved Face', face_img[0])

            cv2.imshow('Video Feed',cv2.flip(frame, 1 ))
            cv2.waitKey(2)
            timer += 5
    else:
        print "This name already exists."
        sys.exit()

def recognize_people(people_folder, shape):
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print "Have you added at least one person to the system?"
        sys.exit()
    '''    
    print "This are the people in the Recognition System:"
    for person in people:
        print "-" + person
    '''
   
    detector = FaceDetector('face_recognition_system/frontal_face.xml')
   
    recognizer = cv2.createLBPHFaceRecognizer()    
    threshold = 105
    
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print "\nOpenCV Error: Do you have at least two people in the database?\n"
        sys.exit()

    video = VideoCamera()

    person_id=raw_input('Enter Your ID:')
    boolean="true"   
    while boolean=="true":
        frame = video.get_frame()
        faces_coord = detector.detect(frame, False)
        if len(faces_coord):
            frame, faces_img = get_images(frame, faces_coord, shape)
            for i, face_img in enumerate(faces_img):
                if __version__ == "3.1.0":
                    collector = cv2.face.MinDistancePredictCollector()
                    recognizer.predict(face_img, collector)
                    conf = collector.getDist()
                    pred = collector.getLabel()
                else:
                    pred, conf = recognizer.predict(face_img)
               #print "Prediction: " + str(pred)
               #print 'Confidence: ' + str(round(conf))
               #print 'Threshold: ' + str(threshold)
                
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.CV_AA)
		    						
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.CV_AA)
                
                cv2.imshow('Video', frame)
                if(labels_people[pred]==str(person_id) and boolean=="true"):                                
                    conn = pyodbc.connect("DRIVER={SQL Server};Server=DESKTOP-M4SA4AV;Database=Ds;uid=DESKTOP-M4SA4AV\fadel;pwd=;Trusted_Connection=yes;")
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM grades where username='"+str(person_id)+"'")
                    results = cursor.fetchone()
                    print("Welcome "+str(labels_people[pred]))
                    while results:    
                        print ("course:" +  str(results[1])+
                               " grade:" +  str(results[2]))  
                        
                        results = cursor.fetchone()
                   
       
            #cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.CV_AA)
        if cv2.waitKey(100) & 0xFF == 27:
            sys.exit()

def check_choice():
    """ Check if choice is good
    """
    is_valid = 0
    while not is_valid:
        try:
            choice = int(raw_input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print "'%d' is not an option.\n" % choice
        except ValueError, error:
            print "%s is not an option.\n" % str(error).split(": ")[1]
    return choice

if __name__ == '__main__':
    print 30 * '-'
    print "   POSSIBLE ACTIONS"
    print 30 * '-'
    print "1. Add person to the recognizer system"
    print "2. Start recognizer"
    print "3. Exit"
    print 30 * '-'

    CHOICE = check_choice()

    PEOPLE_FOLDER = "face_recognition_system/people/"
    SHAPE = "rectangle"

    if CHOICE == 1:
        username=raw_input('username: ')
        password=raw_input('password: ')
        if username=='admin' and password=='admin':
            if not os.path.exists(PEOPLE_FOLDER):
             os.makedirs(PEOPLE_FOLDER)
            add_person(PEOPLE_FOLDER, SHAPE)
        else:
            raw_input("You Don't Have an Access !!")
    elif CHOICE == 2:
        choice="y"
        while choice=='y':
            recognize_people(PEOPLE_FOLDER, SHAPE)
            choice=raw_input('do you want to continue?(y/n)')
    elif CHOICE == 3:
        sys.exit()
