#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import numpy as np
import os
from werkzeug.utils import secure_filename


# In[2]:


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# In[3]:


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


# In[4]:


def detect_and_align_face(image_rgb):
    detector = MTCNN()
    results = detector.detect_faces(image_rgb)
    if results:
        x, y, width, height = results[0]['box']
        margin = 0.2
        x_min, y_min = max(x - int(margin * width), 0), max(y - int(margin * height), 0)
        x_max, y_max = min(x + width + int(margin * width), image_rgb.shape[1]), min(y + height + int(margin * height), image_rgb.shape[0])
        face = image_rgb[y_min:y_max, x_min:x_max]
        aligned_face = cv2.resize(face, (160, 160))
        return aligned_face
    else:
        return None


# In[5]:


def get_face_embedding(face):
    embedder = FaceNet()
    embedding = embedder.embeddings([face])[0]
    return embedding


# In[6]:


def is_same_person(embedding1, embedding2, threshold=0.5):
    distance = cosine(embedding1, embedding2)
    return distance < threshold


# In[ ]:


@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please provide both images'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    image1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename))
    image2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image2.filename))

    image1.save(image1_path)
    image2.save(image2_path)

    image1_rgb = preprocess_image(image1_path)
    image2_rgb = preprocess_image(image2_path)

    face1 = detect_and_align_face(image1_rgb)
    face2 = detect_and_align_face(image2_rgb)

    if face1 is None or face2 is None:
        return jsonify({'result': 'Face not detected in one or both images'}), 400

    embedding1 = get_face_embedding(face1)
    embedding2 = get_face_embedding(face2)

    result = is_same_person(embedding1, embedding2)

    if result:
        return jsonify({'result': 'They are the same person.'})
    else:
        return jsonify({'result': 'They are not the same person.'})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




