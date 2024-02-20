from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
import os
import sqlite3
import numpy as np
import cv2
import base64
import detection
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "123"

con = sqlite3.connect('identifier.sqlite')
con.execute("create table if not exists identifier(pid integer primary key, name text, email text, password text)")
con.close()


# Route for detecting drowsiness
@app.route('/detect_drowsiness', methods=['POST'])
def detect_drowsiness():
    try:
        # Receive image data from the request
        image_data = request.json['image_data']

        # Convert base64 image data to OpenCV format
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect drowsiness
        drowsy = detection.detect_drowsiness(img)

        # Return the result
        return jsonify({'result': drowsy})

    except Exception as E:
        return jsonify({'error': str(E)}), 500


def preprocess_image_for_prediction():
    try:
        # Preprocess image here (resize, normalize, etc.)
        return preprocessed_image
    except Exception as E:
        print(f"Error preprocessing image: {E}")
        return None


# Route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()
    # Preprocess the image data
    processed_image = preprocess_data(image_data)
    # Make prediction
    prediction = model.predict(np.array([processed_image]))
    # Process prediction and return result
    return str(prediction)


@app.route('/', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        con = sqlite3.connect('identifier.sqlite')
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("select * from identifier where email=? and password=?", (email, password))
        data = cur.fetchone()

        if data:
            session["email"] = data["email"]
            session["password"] = data["password"]
            return redirect(url_for("home"))
        else:
            flash("Invalid Email or Password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route('/home', methods=["GET", "POST"])
def home():
    if "email" in session and "password" in session:
        return render_template("home.html")
    else:
        flash("You need to log in first", "danger")
        return redirect(url_for("login"))


@app.route('/register', methods=['GET', 'POST'])
def register():
    global con
    if request.method == 'POST':
        try:
            name = request.form['name']
            password = request.form['password']
            email = request.form['email']
            con = sqlite3.connect('identifier.sqlite')
            cur = con.cursor()
            cur.execute("insert into identifier(name, password, email) values (?, ?, ?)", (name, password, email))
            con.commit()
            flash("Record Added Successfully", "success")
        except Exception as E:
            flash(f"Error in Insert Operation: {E}", "danger")
        finally:
            con.close()
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']
    con = sqlite3.connect('identifier.sqlite')
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from identifier where email=?", (email,))
    user = cur.fetchone()
    con.close()

    if not user:
        return "User not found", 404

    return render_template('profile.html', user=user)


@app.route('/Camera')
def camera():
    if 'email' not in session or 'password' not in session:
        flash("You need to login first", "danger")
        return redirect(url_for("login"))

    return render_template("Camera.html")


# Route for camera feed
@app.route('/video_feed')
def video_feed():
    def generate_frame():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame")
                break

            # Detect drowsiness
            drowsy = detect_drowsiness(frame)

            # Draw text on frame based on drowsiness detection
            if drowsy:
                cv2.putText(frame, 'Drowsy', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Alert', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                app.logger.error("Error encoding frame")
                break

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            cap.release()
            cv2.destroyAllWindows()

        return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/settings')
def settings():
    if 'email' not in session or 'password' not in session:
        flash("You need to login first", "")
        return redirect(url_for("login"))

    return render_template('settings.html')


if __name__ == '__main__':
    app.run(debug=True)
