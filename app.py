import gc
import threading
from functools import wraps

from flask import Flask, render_template, Response, request, flash, redirect, url_for, session

from face_app import generate, get_face_embed_vector, detect_faces_from_cam, cam_open, cam_close, register, \
    return_response, person_recognition, get_image, get_persons_count

# from flask_caching import Cache

# from mediapipe_class import MediaPipeFaceMesh, MediaPipePose
# face_detector_mediapipe = MediaPipeFaceMesh()
# pose_detector_mediapipe = MediaPipePose()
from functions import user_or_admin_insert_logs_for_email

app = Flask(__name__)
app.config["SECRET_KEY"] = "SecretKEy1"


# cache = Cache(app, config={'CACHE_TYPE': 'simple'})


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            # flash('Önce Giriş Yapmalısınız', 'warning')
            return redirect(url_for('login'))

    return wrap


@app.route("/logout")
def logout():
    session.clear()
    flash('Çıkış Yaptınız', 'success')
    gc.collect()
    return redirect(url_for('login'))


@app.route('/login')
def login():
    if 'logged_in' in session:
        return render_template('index.html')
    return render_template("login.html")


@app.route('/')
@app.route('/dashboard')
@login_required
def index():
    person_count = get_persons_count()
    print(session['logged_in'])
    return render_template("index.html", person_count=person_count)


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/open_cam")
def open_cam():
    try:
        cam_open()
        return return_response(message='Kamera Açıldı', status_code=200, code=0)

    except Exception as e:
        return return_response(message='Kamera Açılamadı ' + str(e), status_code=400, code=1)


@app.route("/close_cam")
def close_cam():
    try:
        cam_close()
        return return_response(message='Kamera Kapandı', status_code=200, code=0)
    except Exception as e:
        print("******************")
        return return_response(message='Kamera kapatılamadı ' + str(e), status_code=400, code=1)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route("/test-vector")
def vector_test():
    vector_person_encoded, frame_face, maxAngleFace = get_face_embed_vector()
    print(vector_person_encoded)
    print("************")
    print(frame_face)
    print("----------------")
    print(maxAngleFace)
    return return_response(message=str(vector_person_encoded), status_code=200, code=0)


@app.route("/login-with-button")
def login_button():
    try:
        return person_recognition()
    except Exception as e:
        print(f'Hata Oluştu x {e}')


#
@app.route("/register", methods=["GET", "POST"])
def user_register():
    if request.method == "POST":
        password1 = request.form.get("password")
        password2 = request.form.get("repassword")
        if password1 != password2:
            flash("Şifreler Uyuşmuyor", 'warning')
            return redirect(url_for("user_register"))
        firstName = request.form.get("firstname")
        lastname = request.form.get("lastname")
        # birthdate = request.form.get(py"birthdate")
        user_role = int(request.form.get("user_role"))
        if user_role == 1:
            user_role = "Admin"
        else:
            user_role = "User"
        tcNo = request.form.get("tc")
        email = request.form.get("email")
        data = {"tc": tcNo, "firstname": firstName, "lastname": lastname,
                "user_role": user_role, "email": email,
                "password": password1}

        print(type(user_role))
        # print(firstName, lastname, birthdate, user_role, tcNo, email, password1, password2)
        response = register(data)
        print("asdas")
        print(response.status_code)
        print(response.json['code'])
        if response.status_code == 200:
            if response.json['code'] == 0 or response.json['code'] == 3:
                flash(response.json['message'], 'success')
            if response.json['code'] == 4 or response.json['code'] == 2:
                flash(response.json['message'], 'warning')
            if response.json['code'] == 1:
                flash(response.json['message'], 'warning')
        else:
            flash(
                f'Bir Sorun Oluştu! Hata Kodu : {response.status_code}', 'warning')
            return redirect(url_for('user_register'))
    return render_template('register.html')


@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    return render_template('user_login.html')


@app.route("/login_with_email", methods=["POST"])
def login_with_email():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        print(email)
        response_text, data, response = check_password(email, password)
        if response:
            resp = user_or_admin_insert_logs_for_email(email)
            if not resp:
                flash(message="User log kayıt sorun olutşu", category="error")
                return redirect(url_for('login_with_email'))
            session['logged_in'] = data[0]
            session['logged_in_id'] = data[1]
            return render_template("index.html")
        else:
            flash(response_text, "warning")
    else:
        return redirect(url_for('login'))


def check_password(email, password):
    from MySQL_Connector import connect_to_database
    connection, cursor = connect_to_database()
    cursor.execute("Select user_TC,firstname,lastname,email,password from USERS")
    data = cursor.fetchall()[0]
    name = data[1] + data[2]
    email_t = data[3]
    password_t = data[4]
    if email != email_t:
        return "Email Hatalı", None, False
    elif password != password_t:
        return "Şifre Hatalı", None, False
    else:
        return "Bilgiler Doğru", [name, data[0]], True


@app.route("/kisiler")
def persons():
    from face_app import get_persons
    data = get_persons()
    print(data)
    return render_template("kisiler.html", data=data)


@app.route("/sorgu_yap")
def sorgu_yap():
    resp = person_recognition()
    if resp.json['code'] == 0:
        data = resp.json['result']
        data = data.split("_")
        session['logged_in'] = data[0]
        session['logged_in_id'] = data[1]
        print("************")
        print(data[0])
        print(data[1])
        print("************")

        return return_response(message='SUCCESS', status_code=200, code=0, result=data[0])
    return resp


@app.route("/user_sorgu_yap")
def user_sorgu_yap():
    resp = person_recognition()
    if resp.json['code'] == 0:
        data = resp.json['result']
        data = data.split("_")
        # session['logged_in'] = data[0]
        # session['logged_in_id'] = data[1]
        try:
            from MySQL_Connector import connect_to_database
            cnx, cursor = connect_to_database()
            cursor.execute("select user_TC, user_name, last_login from USERS_LOGS where user_TC=%s", (data[1],))
            log_data = cursor.fetchall()
            cnx.close()
            return return_response(message='SUCCESS', status_code=200, code=0, result=data[0], user_log=log_data)
            # return redirect(url_for('user_login', result=data[0], log_data=log_data))
        except Exception as e:
            print(f'User login log ekleme hata oluştu {e}')
            # return return_response(message='FAILED', status_code=200, code=2, result=data[0])
    return resp


@app.route('/image<string:user_TC>')
def get_image_from_db(user_TC=None):
    return get_image(user_TC)


''' FUNCTIONS OPERATIONS '''

''' FUNCTIONS OPERATIONS '''

if __name__ == '__main__':
    threading.Thread(target=detect_faces_from_cam, daemon=True).start()
    app.run(debug=True, threaded=True, port=3000)
