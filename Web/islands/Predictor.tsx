import { DrawingUtils, FilesetResolver, HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision"
import { useSignal } from "@preact/signals"
import { useEffect, useRef } from "preact/hooks"

export default function Predictor() {
	let handLandmarker: HandLandmarker | null = null
	const camRef = useRef<HTMLVideoElement | null>(null)
	const drawRef = useRef<HTMLCanvasElement | null>(null)
	const workerRef = useRef<Worker | null>(null)
	const ready = useSignal(false)
	const prediction = useSignal<string | null>(null) // Signal for displaying ASL prediction
	let lastVideoTime = -1
	let lastPredictionTime = 0
	const throttleMs = 600 // Predict every 100ms

	const drawLandmarks = (drawingUtils: DrawingUtils, results: HandLandmarkerResult) => {
		if (!results.landmarks) return

		results.landmarks.forEach(landmarks => {
			drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "white", lineWidth: 1 })
			drawingUtils.drawLandmarks(landmarks, { color: "red", radius: 1 })
		})
	}

	const predictWebcam = async () => {
		const cam = camRef.current
		const draw = drawRef.current
		if (!cam || !draw || !handLandmarker || cam.width === 0 || cam.height === 0) return requestAnimationFrame(predictWebcam)

		const ctx = draw.getContext("2d")
		if (!ctx) return

		const startTimeMs = performance.now()
		if (lastVideoTime !== cam.currentTime) {
			lastVideoTime = cam.currentTime
			const results = await handLandmarker.detectForVideo(cam, startTimeMs)

			ctx.clearRect(0, 0, draw.width, draw.height)
			const drawer = new DrawingUtils(ctx)
			drawLandmarks(drawer, results)

			// Throttle and send landmarks to worker for ASL prediction
			if (
				performance.now() - lastPredictionTime > throttleMs
				&& workerRef.current
				&& results.landmarks.length > 0
			) {
				lastPredictionTime = performance.now()
				const dataToSend = results.landmarks.map((landmarks, i) => ({
					landmarks: results.worldLandmarks[i],
					handedness: results.handedness[i][0].index,
				}))
				workerRef.current.postMessage({ landmarks: dataToSend })
			}
		}

		requestAnimationFrame(predictWebcam)
	}

	useEffect(() => {
		const startCamera = async () => {
			try {
				const stream = await navigator.mediaDevices.getUserMedia({ video: true })
				if (camRef.current) camRef.current.srcObject = stream
			} catch (error) {
				console.error("Error accessing camera:", error)
			}
		}

		startCamera()

		return () => {
			if (camRef.current?.srcObject) {
				const stream = camRef.current.srcObject as MediaStream
				stream.getTracks().forEach(track => track.stop())
			}
		}
	}, [])

	useEffect(() => {
		const createHandLandmarker = async () => {
			try {
				const vision = await FilesetResolver.forVisionTasks(
					"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
				)
				handLandmarker = await HandLandmarker.createFromOptions(vision, {
					baseOptions: {
						modelAssetPath:
							"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
						delegate: "GPU",
					},
					runningMode: "VIDEO",
					numHands: 2,
				})
				ready.value = true
				requestAnimationFrame(predictWebcam)
			} catch (error) {
				console.error("Failed to initialize HandLandmarker:", error)
			}
		}
		createHandLandmarker()
	}, [])

	useEffect(() => {
		// Initialize Web Worker
		workerRef.current = new Worker(new URL("/predictionWorker.js", import.meta.url), { type: "module" })

		// Handle messages from worker (e.g., ASL predictions or errors)
		workerRef.current.onmessage = event => {
			const { prediction: pred, error } = event.data
			if (error) {
				console.error("Worker error:", error)
				prediction.value = "Error during prediction"
			} else if (pred !== undefined) {
				prediction.value = `ASL Prediction: ${pred}` // Update UI with prediction
			}
		}

		return () => {
			workerRef.current?.terminate()
		}
	}, [])

	return (
		<main class="flex-1 flex flex-col lg:flex-row gap-4 p-4">
			<div className="flex flex-1 flex-col bg-zinc-700 rounded-xl p-2">
				<div class="relative w-full h-max">
					<video ref={camRef} autoPlay playsInline id="cam" class="w-full h-auto rounded-md" />
					<canvas ref={drawRef} id="draw" class="absolute top-0 w-full h-full" />
				</div>
			</div>
			<div className="flex-1">
				{ready.value
					? (
						<>
							<p>Hand Tracking Active</p>
							<p>{prediction.value || "Waiting for prediction..."}</p>
						</>
					)
					: (
						"Loading Model..."
					)}
			</div>
		</main>
	)
}
