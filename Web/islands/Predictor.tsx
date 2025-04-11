import { DrawingUtils, FilesetResolver, HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision"
import { useSignal } from "@preact/signals"
import { useEffect, useRef } from "preact/hooks"
import { KDTree } from "../utils/KDTree.ts"

export default function Predictor() {
	const mpReady = useSignal(false)
	const camReady = useSignal(false)
	const modelReady = useSignal(false)
	const aslReady = useSignal(true)

	const camRef = useRef<HTMLVideoElement | null>(null)
	const drawRef = useRef<HTMLCanvasElement | null>(null)
	const predictor = useSignal<KDTree | null>(null)
	const prediction = useSignal<string | null>(null)
	let handLandmarker: HandLandmarker | null = null

	const throttleMs = useSignal(1000)
	let lastVideoTime = -1
	let lastPredictionTime = 0

	const history = useSignal<{ timestamp: Date; text: string }[]>([])
	const sentence = useSignal("")

	const processWebcam = () => {
		const cam = camRef.current
		const draw = drawRef.current
		if (!cam || !draw || !handLandmarker) {
			requestAnimationFrame(processWebcam)
			return
		}

		if (cam.videoWidth === 0 || cam.videoHeight === 0) {
			requestAnimationFrame(processWebcam)
			return
		}

		if (draw.width !== cam.videoWidth || draw.height !== cam.videoHeight) {
			draw.width = cam.videoWidth
			draw.height = cam.videoHeight
		}

		const ctx = draw.getContext("2d")
		if (!ctx) {
			requestAnimationFrame(processWebcam)
			return
		}

		const startTimeMs = performance.now()
		if (lastVideoTime !== cam.currentTime) {
			lastVideoTime = cam.currentTime
			const results = handLandmarker.detectForVideo(cam, startTimeMs)

			ctx.clearRect(0, 0, draw.width, draw.height)
			const drawer = new DrawingUtils(ctx)
			drawLandmarks(drawer, results)

			if (results.landmarks.length === 0) {
				prediction.value = null
				lastPredictionTime = performance.now() - throttleMs.value / 4 * 3
				requestAnimationFrame(processWebcam)
				return
			}

			const data = [...results.worldLandmarks[0]?.reduce((acc, cur) => {
				acc.push(cur.x, cur.y, cur.z)
				return acc
			}, [] as number[]), results.handedness[0]?.[0].index]

			if (performance.now() - lastPredictionTime > throttleMs.value && aslReady.value) {
				prediction.value = String.fromCharCode(predict(data) ?? 0)
				if (prediction.value && sentence.value.at(-1) !== prediction.value) sentence.value += prediction.value
				lastPredictionTime = performance.now()
			}
		}

		requestAnimationFrame(processWebcam)
	}

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

	useEffect(() => {
		const startCamera = async () => {
			try {
				const stream = await navigator.mediaDevices.getUserMedia({ video: true })
				if (camRef.current) {
					camRef.current.srcObject = stream
					camReady.value = true
					if (!drawRef.current) return
					drawRef.current.width = camRef.current.videoWidth
				}
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
	}, [drawRef.current, camRef.current])

	useEffect(() => {
		const loadModel = async () => {
			if (predictor.value) return
			const response = await fetch("/model.bin")
			if (!response.ok) throw new Error("Failed to load model")
			const buffer = await response.arrayBuffer()
			const tree = KDTree.deserialize(buffer)
			predictor.value = tree
			modelReady.value = true
		}

		const createHandLandmarker = async () => {
			try {
				const vision = await FilesetResolver.forVisionTasks("")
				handLandmarker = await HandLandmarker.createFromOptions(vision, {
					baseOptions: { modelAssetPath: "/hand_landmarker.task", delegate: "GPU" },
					runningMode: "VIDEO",
					numHands: 1,
					minHandDetectionConfidence: 0.7,
					minHandPresenceConfidence: 0.7,
					minTrackingConfidence: 0.7,
				})
				mpReady.value = true
				requestAnimationFrame(processWebcam)
			} catch (error) {
				console.error("Failed to initialize HandLandmarker:", error)
			}
		}

		loadModel()
		createHandLandmarker()
	}, [])

	// create a keyboard handler for actions

	useEffect(() => {
		const handleKeyDown = (event: KeyboardEvent) => {
			switch (event.key) {
				case "q": {
					sentence.value = sentence.value.replace(/#$/, "").slice(0, -1)
					break
				}

				case "w": {
					sentence.value = sentence.value.replace(/#$/, "").slice(0, -1)
					break
				}

				case "e": {
					sentence.value = ""
					break
				}

				case "r": {
					if (sentence.value.replaceAll("#", "").trim().length === 0) return
					history.value.push({ timestamp: new Date(), text: sentence.value.replaceAll("#", "") })
					sentence.value = ""
					break
				}

				case "space": {
					aslReady.value = !aslReady.value
					if (aslReady.value) {
						prediction.value = null
						lastPredictionTime = performance.now()
					}
					break
				}
			}
		}

		globalThis.addEventListener("keydown", handleKeyDown)

		return () => {
			globalThis.removeEventListener("keydown", handleKeyDown)
		}
	}, [])

	return (
		<main class="w-full h-fit flex-1 flex flex-col lg:flex-row gap-4 p-4 max-w-[2160px] mx-auto">
			<div className="flex lg:w-max h-fit flex-col bg-zinc-700 rounded-3xl gap-2 p-4 lg:max-w-xl sticky top-4">
				<img src="/asl.jpg" alt="ASL Instructions" class="rounded-t-xl rounded-b max-h-64 lg:max-h-max w-auto object-contain" />
				<div class="relative flex justify-center w-full h-auto max-h-64 lg:max-h-max aspect-video bg-zinc-600/50 rounded-t rounded-b-xl overflow-hidden">
					<video ref={camRef} autoPlay playsInline id="cam" class="absolute h-full w-auto" />
					<canvas ref={drawRef} id="draw" class="absolute top-0 h-full" />
				</div>
				<div className="flex flex-col gap-4">
					<div className="flex flex-col gap-1 mt-2 text-zinc-400 px-4">
						<label class="ms-2" htmlFor="speed">Độ trễ nhận diện: {throttleMs.value / 1000}s</label>
						<div className="flex items-center gap-2 justify-between">
							<span class="text-sm tabular-nums">0s</span>
							<input
								type="range"
								id="speed"
								min="0"
								max="2000"
								step="100"
								defaultValue={throttleMs.value.toString()}
								className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
								onInput={e => {
									const value = parseInt((e.target as HTMLInputElement).value)
									throttleMs.value = value
									lastPredictionTime = performance.now()
								}}
							/>
							<span class="text-sm tabular-nums">2s</span>
						</div>
					</div>
					<div className="flex flex-wrap gap-x-4 gap-y-1 ps-2 items-center">
						<div className="flex items-center gap-2">
							<span class={`status ${camReady.value ? "active" : "inactive"}`}></span>Camera
						</div>
						<div className="flex items-center gap-2">
							<span class={`status ${mpReady.value ? "active" : "inactive"}`}></span>Nhận diện
						</div>
						<div className="flex items-center gap-2">
							<span class={`status ${modelReady.value ? "active" : "inactive"}`}></span>Model
						</div>
						<div className="flex items-center gap-2">
							<span class={`status ${aslReady.value ? "active" : ""}`}></span>Dự đoán
						</div>
						<button
							type="button"
							class="rounded-md w-12 px-3 py-1 bg-zinc-600 hover:bg-zinc-500 transition-all outline-none focus:ring focus:ring-zinc-300"
							onClick={() => {
								aslReady.value = !aslReady.value
								if (aslReady.value) {
									prediction.value = null
									lastPredictionTime = performance.now()
								}
							}}
						>
							{aslReady.value ? "⏸" : "▶"}
						</button>
					</div>
				</div>
			</div>
			<div className="flex flex-1 flex-col bg-zinc-700 rounded-3xl gap-2 p-4">
				<div className="w-full fixed left-0 bottom-0 p-2 rounded-t-2xl items-end lg:sticky lg:items-start lg:top-4 lg:p-0 lg:rounded-none flex flex-col gap-2 backdrop-blur-md bg-zinc-700/50">
					<div className="order-last lg:order-first flex gap-2">
						<div key={prediction.value} className="flex rounded-xl justify-center items-center w-28 h-10 bg-zinc-600 animate-flash">
							Dự đoán: {prediction.value || "ø"}
						</div>
						<button type="button"
							class="rounded-xl bg-zinc-600 hover:bg-zinc-500 transition-all outline-none focus:ring focus:ring-zinc-300 px-3 h-10"
							onClick={() => sentence.value = (sentence.value.replace(/[# ]$/, "") + " ").trimStart()}
						>
							␣
						</button>
						<button
							type="button"
							class="rounded-xl bg-zinc-600 hover:bg-zinc-500 transition-all outline-none focus:ring focus:ring-zinc-300 px-3 h-10"
							onClick={() => sentence.value = sentence.value.replace(/#$/, "").slice(0, -1)}
						>
							⌫
						</button>
						<button
							type="button"
							class="rounded-xl bg-zinc-600 hover:bg-zinc-500 transition-all outline-none focus:ring focus:ring-zinc-300 px-3 h-10"
							onClick={() => sentence.value = ""}
						>
							🗑️
						</button>
						<button
							type="button"
							class="rounded-xl bg-zinc-600 hover:bg-zinc-500 transition-all outline-none focus:ring focus:ring-zinc-300 px-3 h-10"
							onClick={() => {
								if (sentence.value.replaceAll("#", "").trim().length === 0) return
								history.value.push({ timestamp: new Date(), text: sentence.value.replaceAll("#", "") })
								sentence.value = ""
							}}
						>
							💾
						</button>
					</div>
					<div className={"min-h-10 w-full flex-1 break-all max-w-full block h-auto rounded-xl bg-zinc-600 px-4 py-2 "
						+ (sentence.value.replaceAll("#", "").length === 0 ? "text-zinc-300" : "")}
					>
						{sentence.value.replaceAll("#", "") || "Aa"}
					</div>
				</div>
				<div className="flex-1 flex flex-col gap-2 max-h-full overflow-scroll">
					{history.value.map(item => (
						<div key={item.timestamp.toString()} className="flex flex-col">
							<span className="ms-2 text-sm text-zinc-400 tabular-nums">
								{item.timestamp.toLocaleTimeString("default", { hour12: false })}
							</span>
							<div className="text-zinc-200 flex-1 w-full max-w-full overflow-x-hidden break-all whitespace-normal">{item.text}</div>
						</div>
					))}
				</div>
			</div>
		</main>
	)
}
