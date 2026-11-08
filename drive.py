
import base64
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi

from flask import Flask
from PIL import Image
from io import BytesIO
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("models", "neurodrive_best.keras")

IMG_HEIGHT = 66
IMG_WIDTH = 200

CROP_TOP = 0.35
CROP_BOTTOM = 0.10

THROTTLE = 0.27
PORT = 4567


# ============================================================
# LOAD MODEL
# ============================================================

model = keras.models.load_model(MODEL_PATH)

print("NeuroDrive model loaded.")


# ============================================================
# SOCKET.IO SERVER
# ============================================================

sio = socketio.Server()

app = Flask(__name__)
app = socketio.Middleware(sio, app)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    h, w = image.shape[:2]

    top = int(h * CROP_TOP)
    bottom = int(h * (1 - CROP_BOTTOM))

    image = image[top:bottom, :, :]

    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Simulator image is already RGB.
    return image.astype(np.uint8)


# ============================================================
# SEND CONTROL COMMANDS
# ============================================================

def send_control(sid, steering, throttle):

    sio.emit(
        "steer",
        data={
            "steering_angle": str(float(steering)),
            "throttle": str(float(throttle))
        },
        room=sid
    )


# ============================================================
# SIMULATOR TELEMETRY
# ============================================================

@sio.on("telemetry")
def telemetry(sid, data):

    if not data or data.get("image") is None:

        send_control(sid, 0.0, 0.0)
        return

    try:

        # Decode simulator camera image
        image_bytes = base64.b64decode(data["image"])

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image = np.asarray(image)

        # Preprocess
        processed = preprocess_image(image)

        # Add batch dimension
        batch = np.expand_dims(processed, axis=0)

        # CNN inference
        steering = float(
            model.predict(batch, verbose=0)[0][0]
        )

        steering = float(
            np.clip(steering, -1.0, 1.0)
        )

        # Fixed throttle for initial testing
        throttle = THROTTLE

        # Send control
        send_control(sid, steering, throttle)

        print(
            f"Steering: {steering:+.3f} | "
            f"Throttle: {throttle:.2f}"
        )

    except Exception as e:

        print("Telemetry error:", e)

        send_control(sid, 0.0, 0.0)


# ============================================================
# SIMULATOR CONNECTION
# ============================================================

@sio.on("connect")
def connect(sid, environ):

    print("Simulator connected!")

    send_control(sid, 0.0, 0.0)


@sio.on("disconnect")
def disconnect(sid):

    print("Simulator disconnected.")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(f"Starting NeuroDrive server on port {PORT}...")

    eventlet.wsgi.server(
        eventlet.listen(("", PORT)),
        app
    )
