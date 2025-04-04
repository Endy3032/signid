import { DrawingUtils, FilesetResolver, HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision"
import { useSignal } from "@preact/signals"
import { useEffect, useRef } from "preact/hooks"
import { KDTree } from "../utils/KDTree.ts"

export default function Predictor() {
	const ready = useSignal(false)
	const camRef = useRef<HTMLVideoElement | null>(null)
	const drawRef = useRef<HTMLCanvasElement | null>(null)
	const predictor = useSignal<KDTree | null>(null)
	const prediction = useSignal<string | null>(null)
	let handLandmarker: HandLandmarker | null = null
	let lastVideoTime = -1

	const predict = (landmarks: number[]) => {
		if (!predictor.value) return null
		const f32 = new Float32Array(landmarks)
		const predictions = predictor.value?.query(f32)
		return predictions
	}

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
		if (!cam || !draw || !handLandmarker) {
			requestAnimationFrame(predictWebcam)
			return
		}

		if (cam.videoWidth === 0 || cam.videoHeight === 0) {
			requestAnimationFrame(predictWebcam)
			return
		}

		if (draw.width !== cam.videoWidth || draw.height !== cam.videoHeight) {
			draw.width = cam.videoWidth
			draw.height = cam.videoHeight
		}

		const ctx = draw.getContext("2d")
		if (!ctx) {
			requestAnimationFrame(predictWebcam)
			return
		}

		const startTimeMs = performance.now()
		if (lastVideoTime !== cam.currentTime) {
			lastVideoTime = cam.currentTime
			const results = await handLandmarker.detectForVideo(cam, startTimeMs)

			ctx.clearRect(0, 0, draw.width, draw.height)
			const drawer = new DrawingUtils(ctx)
			drawLandmarks(drawer, results)

			if (results.landmarks.length === 0) {
				prediction.value = "No hands detected"
				requestAnimationFrame(predictWebcam)
				return
			}

			const data = [...results.worldLandmarks[0]?.reduce((acc, cur) => {
				acc.push(cur.x, cur.y, cur.z)
				return acc
			}, [] as number[]), results.handedness[0]?.[0].index]

			prediction.value = String.fromCharCode(predict(data) ?? 0)
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

		const loadModel = async () => {
			if (predictor.value) return
			const response = await fetch("/model.bin")
			if (!response.ok) throw new Error("Failed to load model")
			const buffer = await response.arrayBuffer()
			const tree = KDTree.deserialize(buffer)
			predictor.value = tree
		}

		const createHandLandmarker = async () => {
			try {
				const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm")
				handLandmarker = await HandLandmarker.createFromOptions(vision, {
					baseOptions: {
						modelAssetPath:
							"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
						delegate: "GPU",
					},
					runningMode: "VIDEO",
					numHands: 1,
				})
				ready.value = true
				requestAnimationFrame(predictWebcam)
			} catch (error) {
				console.error("Failed to initialize HandLandmarker:", error)
			}
		}

		startCamera()
		loadModel()
		createHandLandmarker()

		return () => {
			if (camRef.current?.srcObject) {
				const stream = camRef.current.srcObject as MediaStream
				stream.getTracks().forEach(track => track.stop())
			}
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
