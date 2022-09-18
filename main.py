# Importar librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

# Creamos nuestra funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

# Realizar nuestra VideoCaptura
cap = cv2.VideoCapture(0)

# Funcion frames
def gen_frame():
    #Empezamos
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        else:
            # Correccion de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Observamos los resultados
            resultados = MallaFacial.process(frameRGB)

            # Si tenemos rostros
            if resultados.multi_face_landmarks:
                # Iteramos
                for rostros in resultados.multi_face_landmarks:
                    # Dibujamos
                    mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()

        yield (b'--frame\r\n'
               b'content-type: image/jpeg\r\n\r\n' + frame)

# Creamos la app
app = Flask(__name__)

# Ruta principal
@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar nuestra web
if __name__ == "__main__":
    app.run(debug = True)