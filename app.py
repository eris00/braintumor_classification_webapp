from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# učitavanje modela:
model = load_model('model/brainTumor_model_test.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # Postavljanje direktorijuma gdje će se čuvati uploadane slike
        upload_folder = 'uploads'
        app.config['UPLOAD_FOLDER'] = upload_folder

        # Dobijanje uploadane slike
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = f"{upload_folder}/{uploaded_file.filename}"
            uploaded_file.save(img_path)

            # Učitavanje i obrada slike za predikciju
            img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predikcija
            predictions = model.predict(img_array)

            # Postavljanje rezultata
            if predictions[0, 0] > 0.5:
                result = "Tumor"
            else:
                result = "Healthy"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)