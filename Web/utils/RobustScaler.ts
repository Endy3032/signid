import { dirname, fromFileUrl } from "@std/path"

export class RobustScaler {
	private medians: Float32Array | null = null
	private iqrs: Float32Array | null = null

	/**
	 * Calculate the median of a sorted array
	 * @param sorted - Sorted array
	 * @returns The median value
	 */
	private median(sorted: number[]): number {
		const mid = Math.floor(sorted.length / 2)
		return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
	}

	/**
	 * Calculate the quartile of a sorted array
	 * @param sorted - Sorted array
	 * @param q - Quartile (0.25, 0.5, 0.75)
	 * @returns The value at the specified quartile
	 */
	private quartile(sorted: number[], q: 0.25 | 0.5 | 0.75): number {
		const pos = (sorted.length - 1) * q
		const base = Math.floor(pos)
		const rest = pos - base
		if (sorted[base + 1] === undefined) return sorted[base]
		return sorted[base] + rest * (sorted[base + 1] - sorted[base])
	}

	/**
	 * Fit the scaler to the data
	 * @param data - Input data
	 * @returns The fitted scaler
	 * @throws Error if the input data is empty
	 */
	fit(data: number[][]) {
		if (data.length === 0) throw new Error("Input data X is empty")
		const dim = data[0].length

		this.medians = new Float32Array(dim)
		this.iqrs = new Float32Array(dim)

		for (let i = 0; i < dim; i++) {
			const column = data.map(row => row[i])
			const sorted = column.toSorted((a, b) => a - b)
			this.medians[i] = this.median(sorted)
			const q1 = this.quartile(sorted, 0.25)
			const q3 = this.quartile(sorted, 0.75)
			this.iqrs[i] = (q3 - q1) || 1
		}

		return this
	}

	/**
	 * Transform the data using the fitted scaler
	 * @param X - Input data
	 * @returns Scaled data
	 * @throws Error if the input data is empty
	 * @throws Error if the scaler is not fitted
	 * @throws Error if the feature count of the input data does not match the fitted scaler
	 */
	transform(X: number[][]) {
		if (!this.medians || !this.iqrs) throw new Error("Scaler not fitted")
		const p = this.medians.length

		return X.map(row => {
			if (row.length !== p) throw new Error("Feature count mismatch")
			return row.map((value, j) => (value - this.medians![j]) / this.iqrs![j])
		})
	}

	/**
	 * Fit the scaler to the data and transform it
	 * @param X - Input data
	 * @returns Scaled data
	 * @throws Error if the input data is empty
	 * @throws Error if the scaler is not fitted
	 * @throws Error if the feature count of the input data does not match the fitted scaler
	 */
	fitTransform(X: number[][]) {
		return this.fit(X).transform(X)
	}

	/**
	 * Serialization to binary
	 * Structure:
	 * - 4 bytes for dimension -> 4 bytes (uint32)
	 * - 4 bytes for each median -> 4 * dimension bytes (float32array)
	 * - 4 bytes for each IQR -> 4 * dimension bytes (float32array)
	 */
	serialize() {
		if (!this.medians || !this.iqrs) throw new Error("Scaler not fitted")
		const buffer = new ArrayBuffer(4 + 4 * this.medians.length * 2)
		const view = new DataView(buffer)
		view.setUint32(0, this.medians.length, true)

		for (let i = 0; i < this.medians.length; i++) {
			view.setFloat32(4 + i * 4, this.medians[i], true)
			view.setFloat32(4 + (i + this.medians.length) * 4, this.iqrs[i], true)
		}

		return buffer
	}

	/**
	 * Save the scaler to a file
	 * @param filePath - Path to the file
	 */
	async save(filePath: string) {
		const buffer = this.serialize()
		await Deno.writeFile(filePath, new Uint8Array(buffer))
	}

	/**
	 * Deserialization from binary
	 * @param buffer - Binary data
	 */
	static deserialize(buffer: ArrayBuffer) {
		const view = new DataView(buffer)
		const dim = view.getUint32(0, true)
		const medians = new Float32Array(dim)
		const iqrs = new Float32Array(dim)

		for (let i = 0; i < dim; i++) {
			medians[i] = view.getFloat32(4 + i * 4, true)
			iqrs[i] = view.getFloat32(4 + (i + dim) * 4, true)
		}

		const scaler = new RobustScaler()
		scaler.medians = medians
		scaler.iqrs = iqrs
		return scaler
	}

	/**
	 * Load the scaler from a file
	 * @param filePath - Path to the file
	 */
	static async load(filePath: string) {
		const buffer = await Deno.readFile(filePath)
		return RobustScaler.deserialize(buffer.buffer)
	}
}

if (import.meta.main) {
	Deno.chdir(fromFileUrl(dirname(import.meta.url)))

	const X = [
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9],
		[10, 11, 12],
	]

	const scaler = new RobustScaler()
	const X_scaled = scaler.fitTransform(X)
	console.log("Scaled data:", X_scaled)
	console.log("Scaler:", scaler)
	const serialized = scaler.serialize()
	console.log("Serialized data:", serialized)
	scaler.save("scaler.bin")
	const loadedScaler = await RobustScaler.load("scaler.bin")
	console.log("Loaded scaler:", loadedScaler)
}
