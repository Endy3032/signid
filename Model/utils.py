import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
from typing import Literal

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


def drawLandmarks(rgbImage, detectionResult, mode: Literal["overlay", "transparent"] = "overlay", predictions=[]):
	h, w, _ = rgbImage.shape
	landmarksList = detectionResult.hand_landmarks
	handednessList = detectionResult.handedness
	output = np.zeros(rgbImage.shape, dtype=np.uint8) if mode == "transparent" else np.copy(rgbImage)

	# Loop through detected hands
	for idx in range(len(landmarksList)):
		landmarks = landmarksList[idx]
		handedness = handednessList[idx]

		# Draw landmarks
		normalizedLandmarks = NormalizedLandmarkList()
		normalizedLandmarks.landmark.extend([NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks])
		solutions.drawing_utils.draw_landmarks(
			output,
			normalizedLandmarks,
			solutions.hands.HAND_CONNECTIONS,
			solutions.drawing_styles.get_default_hand_landmarks_style(),
			solutions.drawing_styles.get_default_hand_connections_style(),
		)

		x = [landmark.x for landmark in landmarks]
		y = [landmark.y for landmark in landmarks]
		textX = int(min(x) * w)
		textY = int(min(y) * h) - 20

		# Draw handedness
		cv2.putText(
			img=output,
			text=f"{handedness[0].category_name}",
			org=(textX, textY),
			fontFace=cv2.FONT_HERSHEY_DUPLEX,
			fontScale=1,
			color=(0, 0, 0),
			thickness=5,
			lineType=cv2.LINE_AA,
		)
		cv2.putText(
			img=output,
			text=f"{handedness[0].category_name}",
			org=(textX, textY),
			fontFace=cv2.FONT_HERSHEY_DUPLEX,
			fontScale=1,
			color=(255, 255, 255),
			thickness=2,
			lineType=cv2.LINE_AA,
		)

	return output


def draw_landmarks_on_image(rgb_image, detection_result, predictions):
	h, w, _ = rgb_image.shape
	hand_landmarks_list = detection_result.hand_landmarks
	annotated_image = np.zeros(rgb_image.shape, dtype=np.uint8)

	# Loop through the detected hands to visualize.
	for idx in range(len(hand_landmarks_list)):
		hand_landmarks = hand_landmarks_list[idx]

		hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
		])

		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			hand_landmarks_proto,
			solutions.hands.HAND_CONNECTIONS,
			solutions.drawing_styles.get_default_hand_landmarks_style(),
			solutions.drawing_styles.get_default_hand_connections_style(),
		)

		x = max([landmark.x for landmark in hand_landmarks])
		y = min([landmark.y for landmark in hand_landmarks])
		text_x = int(x * w) + MARGIN
		text_y = int(y * h) - MARGIN

		annotated_image = cv2.flip(annotated_image, 1)
		cv2.putText(
			annotated_image,
			predictions[idx],
			(w - text_x, text_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			FONT_SIZE,
			HANDEDNESS_TEXT_COLOR,
			FONT_THICKNESS,
			cv2.LINE_AA,
		)
		annotated_image = cv2.flip(annotated_image, 1)

	return annotated_image


def add_transparent_image(background, foreground):
	alpha = np.uint8(np.sum(foreground, axis=-1) > 0)
	foreground = np.dstack((foreground, alpha))

	foreground_colors = foreground[:, :, :3]
	alpha_mask = alpha[:, :, np.newaxis]

	return background * (1 - alpha_mask) + foreground_colors * alpha_mask
