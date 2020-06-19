from flask import Flask,render_template,request
from PIL import Image
from werkzeug.utils import secure_filename
import os

import trafficmodel
from trafficmodel import load_model,load_transforms,predict

MODEL_PATH='models/model.pt'
UPLOAD_FOLDER='static/'
ALLOWED_EXTENSIONS=set(['png','jpg','jpeg'])

app=Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

model=load_model(MODEL_PATH)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input',methods=['POST','GET'])
def upload():
    if request.method=='POST':
        img_file=request.files['image_name']
        filename=secure_filename(img_file.filename)
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        image=Image.open(img_file)
        preprocess=load_transforms(image)

        label=predict(model,preprocess,image)
        return render_template('result.html',label=label,filename=filename)
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(port=1111,debug=True)
    
