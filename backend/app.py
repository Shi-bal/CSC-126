# Just run this file to start the web app
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from detect import run_pipeline, load_snake_data

UPLOAD_FOLDER = "static/uploads"

app = Flask(__name__)
app.secret_key = b'\x0b\\ \xcb\x8c\x1f\xf7\xc4\x9dx\xe4`\x1c\xed);/\x86\xf1f4\x93\xb4\xae'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snake_identifier')
def show_upload_page():
    return render_template('snake_identifier.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    rel_result_path, predicted_class = run_pipeline(file_path, filename)

    snake_data = load_snake_data()
    snake_info = snake_data.get(predicted_class, {
        "name": predicted_class,
        "scientific_name": "Unknown",
        "family": "Unknown",
        "description": "No description available.",
        "danger": "Unknown",
        "fangs": "Unknown",
        "length": "Unknown",
        "rarity": "Unknown"
    })

    # Save results in session
    session['result_path'] = rel_result_path
    session['snake_info'] = snake_info

    # Redirect to result page route
    return redirect(url_for('show_result'))

@app.route('/result')
def show_result():
    # Get the info from session
    result_path = session.get('result_path')
    snake_info = session.get('snake_info')

    if not result_path or not snake_info:
        # If no info, redirect to upload page
        return redirect(url_for('show_upload_page'))

    return render_template("snake_identifier_result.html", image=result_path, info=snake_info)

if __name__ == '__main__':
    app.run(debug=True)
