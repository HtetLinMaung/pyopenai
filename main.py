import urllib

from flask import Flask, request
import face_recognition

BASE_URL = '/pyopenai'

app = Flask(__name__)


def load_image_url(url):
    response = urllib.request.urlopen(url)
    return face_recognition.load_image_file(response)


@app.route(BASE_URL)
def welcome():
    return """<div style="height: 100vh; display: flex; justify-content: center; align-items: center;">
    <h1>Python OpenAI Server Online</h1>
    </div>"""


@app.route(f"{BASE_URL}/face-detector/detect-faces", methods=['POST'])
def detect_faces():
    try:
        data = []
        faces = request.json['faces']
        model = request.json['model']

        for url in faces:
            image = load_image_url(url)
            face_locations = face_recognition.face_locations(image, model=model)
            data.append(face_locations)

        return {'code': 200, 'message': 'Detecting faces successful', 'data': data}
    except Exception as err:
        print(err)
        return {'code': 500, 'message': 'Internal Server Error'}, 500


@app.route(f"{BASE_URL}/face-detector/identify-faces", methods=['POST'])
def identify_faces():
    try:
        data = []
        known_encodings = []
        known_labels = []

        known_faces = request.json['known_faces']
        unknown_faces = request.json['unknown_faces']
        tolerance = request.json['tolerance']
        model = request.json['model']

        print('loading known faces')
        for face in known_faces:
            image = load_image_url(face['url'])
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_labels.append(face['label'])

        print('processing unknown faces')
        for url in unknown_faces:
            image = load_image_url(url)
            locations = face_recognition.face_locations(image, model=model)
            encodings = face_recognition.face_encodings(image, locations)

            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_encodings,
                                                         face_encoding,
                                                         tolerance=tolerance)

                if True in results:
                    data.append({
                        'match': known_labels[results.index(True)],
                        'face_location': face_location,
                        'top_left': (face_location[3], face_location[0]),
                        'bottom_right': (face_location[1], face_location[2])
                    })

        return {'code': 200, 'message': 'Identifying faces successful', 'data': data}
    except Exception as err:
        print(err)
        return {'code': 500, 'message': 'Internal Server Error'}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)
