import os
import json
import logging
import re

from flask import Flask, Response, current_app, render_template, request, redirect, flash, send_from_directory
from flask_cors import CORS
from flask_session import Session  # https://pythonhosted.org/Flask-Session

from werkzeug.utils import secure_filename
import cv2
import numpy as np

import openai
from azure.identity import DefaultAzureCredential

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER')
ALLOWED_EXTENSIONS = os.environ.get('ALLOWED_EXTENSIONS')

# Azure OpenAI
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL = os.environ.get("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")

# set logging level
logging.basicConfig(level=logging.WARNING)

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed,
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
# keys for each service
# If you encounter a blocking error during a DefaultAzureCredntial resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
azure_credential = DefaultAzureCredential()

openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = AZURE_OPENAI_API_VERSION
if AZURE_OPENAI_KEY is None:
    openai.api_type = "azure_ad"
    openai_token = azure_credential.get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    openai.api_key = openai_token.token
else:
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_KEY


app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/compare-image", methods=['GET', 'POST'])
def compare_image():
    if request.method == "GET":
        return render_template("tools.html")
    else:
        # check if the post request has the file part
        if 'image1' not in request.files:
            flash('Not found file image1')
            return redirect(request.url)
        if 'image2' not in request.files:
            flash('No found file image2')
            return redirect(request.url)
        
        image1 = request.files['image1']
        image2 = request.files['image2']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if image1.filename == '':
            flash('No selected image1')
            return redirect(request.url)
        if image2.filename == '':
            flash('No selected image2')
            return redirect(request.url)
        
        if image1 and allowed_file(image1.filename):
            image1_filename = secure_filename(image1.filename)
            image1_path = os.path.join(UPLOAD_FOLDER, image1_filename)
            image1.save(image1_path)
        if image2 and allowed_file(image1.filename):
            image2_filename = secure_filename(image2.filename)
            image2_path = os.path.join(UPLOAD_FOLDER, image2_filename)
            image2.save(image2_path)

        # Read the images using OpenCV
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # Get the dimensions of the first image
        height1, width1, _ = image1.shape

        # Resize the second image to match the dimensions of the first image
        image2_resized = cv2.resize(image2, (width1, height1))

        image2 = image2_resized

        # Compare the images using the Structural Similarity Index (SSIM)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        error, diff1, diff2 = mse(gray1, gray2)

        print("Image matching Error between the two images:", error)
        diff1_file_name = 'diff_img_1.jpg'
        diff1_file_path = os.path.join(UPLOAD_FOLDER, diff1_file_name)
        cv2.imwrite(diff1_file_path, diff1)
        
        diff2_file_name = 'diff_img_2.jpg'
        diff2_file_path = os.path.join(UPLOAD_FOLDER, diff2_file_name)
        cv2.imwrite(diff2_file_path, diff2)

        return render_template('tools.html', result_image_1 = diff1_file_name, result_image_2 = diff2_file_name)

# define the function to compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    # Both images should be of same depth and type. Note that when used with RGBA images, the alpha channel is also subtracted.
    diff1 = cv2.subtract(img1, img2)
    diff2 = cv2.subtract(img2, img1)
    err = np.sum(diff1**2)
    mse = err/(float(h*w))
    return mse, diff1, diff2

@app.get('/image-data/<path:filename>')
def download(filename):
    folder_path = os.path.join(current_app.root_path, UPLOAD_FOLDER)
    return send_from_directory(folder_path, filename)



@app.route('/text-assistant', methods=['GET', 'POST'])
def text_assistant():
    try:
        if request.method == "GET":
            return render_template("text-assistant.html")
        else:
            data = request.json
            user_content = data['user_content']
            language = data['language']
            option = data['option']
            style = data['style']
            system_content = 'You will be provided with statements, and your task is to '
            # convert them to standard ' + data['language'] + '.'
            
            if option == 'translate':
                system_content += 'translate them'
            elif option == 'explain':
                system_content += 'explain them for a second-grade student'
            elif option == 'main-ideas':
                system_content += 'list the main ideas of them'
            elif option == 'summarize':
                system_content += 'summarize them in one sentence'
            else:
                system_content += 'convert them'

            if style == 'standard':
                system_content += ' to standard ' + language + '.'
            elif style == 'friendly':
                system_content += ' to ' + language + ' in a friendly style.'
            elif style == 'humorous':
                system_content += ' to ' + language + ' in a humorous style.'
            elif style == 'solid':
                system_content += ' to ' + language + ' in a solid style.'
            elif style == 'lovely':
                system_content += ' to ' + language + ' in a lovely style.'
            else:
                system_content += ' to standard ' + language + '.'



            response = openai.ChatCompletion.create(engine = AZURE_OPENAI_CHAT_DEPLOYMENT, 
                                                messages=[
                                                    {"role": "system", "content": system_content},
                                                    {"role": "user", "content": user_content}
                                                ],
                                                temperature = 0.3, 
                                                max_tokens = 400)
            return response.choices[0].message
        
    except Exception as e:
        print(e.args)
        return json.dumps({'error': e.args})

@app.route("/<path:path>")
def static_file(path):
    return app.send_static_file(path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
