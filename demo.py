import math
import joblib
import mediapipe as mp
import cv2
import numpy as np
from utils import draw_landmarks_on_image, add_transparent_image

with open("archive/predictor_v1.pkl", "rb") as f:
	classifier = joblib.load(f)

with open("archive/scaler_v1.pkl", "rb") as f:
	scaler = joblib.load(f)

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

predictions = []
drawn_frame = np.array([])


def print_result(result, output_image, timestamp_ms):
	try:
		landmarks_ls = result.hand_world_landmarks
		handedness_ls = result.handedness
		global predictions
		global drawn_frame
		predictions = []

		for idx in range(len(landmarks_ls)):
			landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks_ls[idx]]).flatten()
			handedness = handedness_ls[idx][0].index

			data = scaler.transform(np.array([[handedness] + landmarks.tolist()]))
			prediction = classifier.predict(data)
			predictions.append(prediction[0])

		drawn_frame = draw_landmarks_on_image(output_image.numpy_view(), result, predictions)
	except Exception as e:
		print(e)


options = mp.tasks.vision.HandLandmarkerOptions(
	base_options=mp.tasks.BaseOptions(model_asset_path="./models/hand_landmarker.task"),
	running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
	num_hands=2,
	result_callback=print_result,
	min_hand_detection_confidence=0.5,
	min_hand_presence_confidence=0.5,
	min_tracking_confidence=0.5,
)

cam = cv2.VideoCapture(0)
timestamp = 0
show_timestamp = 0
INTERVAL = 10
fps = math.floor(1000 / cam.get(cv2.CAP_PROP_FPS))
print(cam.get(cv2.CAP_PROP_FPS))
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

with HandLandmarker.create_from_options(options) as landmarker:
	while cam.isOpened():
		ret, frame = cam.read()
		if not ret:
			print("Dead")
			break

		timestamp += fps
		mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if timestamp - show_timestamp > INTERVAL:
			landmarker.detect_async(mp_img, timestamp)
			show_timestamp = timestamp

		if drawn_frame.size > 0:
			frame = add_transparent_image(frame, drawn_frame)

		cv2.imshow("Camera", cv2.flip(frame, 1))

		if cv2.waitKey(5) & 0xFF == 27:
			break

cam.release()
cv2.destroyAllWindows()
