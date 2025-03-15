import * as ort from "/ort.min.mjs"

let onnxSession = null

self.onmessage = async event => {
	const { landmarks } = event.data

	if (!onnxSession) {
		ort.env.wasm.numThreads = 1
		ort.env.wasm.wasmPaths = {
			wasm: "/ort-wasm-simd-threaded.wasm",
			mjs: "/ort-wasm-simd-threaded.mjs",
		}

		try {
			onnxSession = await ort.InferenceSession.create("/model.onnx", { executionProviders: ["wasm"] })
			console.log("ONNX model loaded in worker")
		} catch (error) {
			self.postMessage({ error: error })
			return
		}
	}

	try {
		for (const { landmarks: worldLandmarks, handedness } of landmarks) {
			const input = new Float32Array(64)
			input[0] = handedness
			for (let j = 0; j < 21; j++) {
				const x = worldLandmarks[j].x
				const y = worldLandmarks[j].y
				const z = worldLandmarks[j].z
				input[1 + j * 3] = x
				input[1 + j * 3 + 1] = y
				input[1 + j * 3 + 2] = z
			}

			const inputTensor = new ort.Tensor("float32", input, [1, 64])
			const outputMap = await onnxSession.run({ input: inputTensor }, ["output_label"])
			const output = outputMap["output_label"]

			self.postMessage({ prediction: output.data[0] })
		}
	} catch (error) {
		self.postMessage({ error })
	}
}
