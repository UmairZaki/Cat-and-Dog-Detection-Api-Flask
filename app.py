from flask import Flask,jsonify,request,render_template,request
import  json
import os
import numpy as np
import keras.backend.tensorflow_backend as tb
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

loaded_model = load_model('cats_and_dogs_small_2.h5')
@app.route("/show_image",methods=['POST'])
def Home():
    img = request.files.get('data').read()
    stream = BytesIO(img)
    image = Image.open(stream).convert("RGBA")
    stream.close()
    photo = np.array(image)
    x = np.resize(photo, (150,150,3))


    result = loaded_model.predict_classes(x.reshape(1,150,150,3))
    
    if result == 1:
        return render_template('x.html',r = 'Dog')
    elif result == 0:
        return render_template('x.html',r = 'Cat')
   
    
@app.route("/")
def Upload_image():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False,threaded=False)