from flask import Flask,render_template, request
#from flask_mysqldb import MySQL
# import mysql.connector
app = Flask(__name__)
import pymysql 
import cv2
import numpy as np
import pytesseract
import re
from pytesseract import Output
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'admin'
# app.config['MYSQL_DB'] = 'data'
 
# mysql = MySQL(app)
 
@app.route('/')
def form():
    return render_template('home.html')
 
@app.route('/reg', methods = ['POST', 'GET'])
def reg():
    if request.method == 'GET':
        return "Login via the login Form"
     
    if request.method == 'POST':
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
        tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
        img = request.files['image']
        file_name = "static\\upload"
        UPLOADS_PATH = join(dirname(realpath(__file__)), file_name)
        img.save(os.path.join(UPLOADS_PATH, secure_filename(img.filename)))
        # default_storage.save(file_name, img)
        print(img.filename)
        imgpath=str(img.filename)
        path='./static/upload/'+imgpath
        img1 =cv2.imread(path)
        rgb_planes = cv2.split(img1)
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

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 7)
        # cv2.imshow('croppicds', faces)
        for (x, y, w, h) in faces:
            ix = 0
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 0), 2)
            roi_color = img1[y:y + h, x:x + w]
            # crop_pic = cv2.imwrite('croppic10.jpg', roi_color)
            # crop_pic = cv2.imshow('croppicds', roi_color)
            path='./static/crop/'+imgpath
            cv2.imwrite(path, roi_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # path='static/crop'
        # UPLOADS_PATH = join(dirname(realpath(__file__)), path)
        # crop_pic.save(os.path.join(UPLOADS_PATH, secure_filename(img.filename)))
        # path='./static/crop/'+imgpath
            
            

        conn = pymysql.connect(
        host='localhost',
        user='root', 
        password = "admin",
        db='card',
        )
        cur = conn.cursor()

        sql = "INSERT INTO idcard (name,date,sex,number) VALUES (%s,%s,%s,%s)"
        val = (name,date,sex,num)
        cur.execute(sql, val)
        # cur.execute('SELECT * FROM student WHERE name = %s AND id = %s', (name, id,))
        # Fetch one record and return result
        account = cur.fetchone()
        conn.commit()  
        return render_template('datas.html',date=date,num=num,name=name,sex=sex,path=path)

@app.route('/reg', methods = ['POST', 'GET'])
def reg():
    from utils import visualization_utils as vis_util
    from utils import label_map_util
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import sys
    from PIL import Image

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'model'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3, 1280)
    ret = video.set(4, 720)

    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame1 = video.read()
        frame=frame1
        frame_expanded = np.expand_dims(frame1, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        print(np.squeeze(boxes))
        # Draw the results of the detection (aka 'visulaize the results')
        image, array_coord=vis_util.visualize_boxes_and_labels_on_image_array(
            frame1,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        ymin, xmin, ymax, xmax = array_coord
        shape = np.shape(image)
        im_width, im_height = shape[1], shape[0]
        (left, right, top, bottom) = (xmin * im_width,
                                    xmax * im_width, ymin * im_height, ymax * im_height)

        # Using Image to crop and save the extracted copied image
        path='./static/id/crop.jpg'
        cv2.imwrite(path, frame)
        output_path = 'test_images/output/output.png'
        # im = Image.open(path)
        # im.crop((left, top, right, bottom)).save(output_path, quality=95)
        roi = frame[top:bottom, left:right]
        cv2.imwrite(output_path, roi)

        # image_cropped = cv2.imread(output_path)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('ID CARD DETECTOR', frame)
        # count = 0
        # for i in scores:
        #     count += 1
        # print("Count", count)
    #    print("Scores" ,scores)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()
                
        
app.run(host='localhost', port=5000,debug=True)