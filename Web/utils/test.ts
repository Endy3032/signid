import { parse } from "@std/csv"
import { KDTree, KDTreePoint } from "./KDTree.ts"

async function loadCSV(filePath: string): Promise<KDTreePoint[]> {
	const csvText = await Deno.readTextFile(filePath)
	const rows = parse(csvText, { skipFirstRow: true }) // Skip header

	const data: KDTreePoint[] = []
	for (const row of rows) {
		const point = new Float32Array(64)
		const values = Object.values(row)
		// x0 to z20 (63 coords, columns 2-64)
		for (let i = 0; i < 63; i++) {
			point[i] = parseFloat(values[i + 2] ?? "")
		}
		// Handedness (dim 63, column 1)
		point[63] = parseInt(values[1] ?? "") // hand
		const label = values[0]?.charCodeAt(0) ?? 65
		data.push([point, label])
	}
	return data
}

async function testKDTree() {
	console.log("--- KDTree Test ---")
	console.log("Loading train.csv...")
	const trainData = await loadCSV("train.csv")
	console.log(`Train size: ${trainData.length}`)

	console.log("Building KD-tree...")
	const tree = new KDTree()
	tree.build(trainData)

	console.log("Loading test.csv...")
	const testData = await loadCSV("test.csv")
	console.log(`Test size: ${testData.length}`)

	console.log("Testing...")
	let correct = 0
	for (const [point, trueLabel] of testData) {
		const predictedLabel = tree.query(point, 3, 10)
		if (predictedLabel === trueLabel) correct++
		if (String.fromCharCode(trueLabel) === String.fromCharCode(predictedLabel)) continue
		console.log(`False: ${String.fromCharCode(trueLabel)}/${String.fromCharCode(predictedLabel)}`)
	}

	let accuracy = (correct / testData.length) * 100
	console.log(`Accuracy: ${accuracy.toFixed(2)}% (${correct}/${testData.length})`)

	console.log("--- Serialization Test ---")
	const serializedKDTree = tree.serialize()
	await Deno.writeFile("./model.bin", new Uint8Array(serializedKDTree))
	const serializedModel = await Deno.readFile("./model.bin")
	const deserializedKDTree = KDTree.deserialize(serializedModel.buffer)

	console.log("Testing...")
	correct = 0
	for (const [point, trueLabel] of testData) {
		const predictedLabel = deserializedKDTree.query(point, 3, 10)
		if (predictedLabel === trueLabel) correct++
		if (String.fromCharCode(trueLabel) === String.fromCharCode(predictedLabel)) continue
		console.log(`False: ${String.fromCharCode(trueLabel)}/${String.fromCharCode(predictedLabel)}`)
	}

	accuracy = (correct / testData.length) * 100
	console.log(`Accuracy: ${accuracy.toFixed(2)}% (${correct}/${testData.length})`)
}

await testKDTree()
