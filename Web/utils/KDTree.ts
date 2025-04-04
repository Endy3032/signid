import { SimpleMaxHeap } from "./MaxHeap.ts"

/**
 * KD-Tree Structure for ASL Detection
 * - Purpose: Stores scaled 64D hand landmark vectors (63D from MediaPipe + 1D handedness)
 *   with ASCII labels for k=3 nearest neighbor classification.
 * - Node:
 *   - point: Float32Array(64) - Scaled [x1, y1, z1, ..., x21, y21, z21, handedness]
 *   - label: uint8 - ASCII value (e.g., 65 for "A", ..., 90 for "Z")
 *   - splitDim: uint8 - Dimension to split on (0-63)
 *   - splitValue: float32 - Value in point[splitDim] defining the split
 *   - left: KDTreeNode | null - Subtree for points < splitValue in splitDim
 *   - right: KDTreeNode | null - Subtree for points >= splitValue in splitDim
 * - Tree:
 *   - root: KDTreeNode | null - Top node
 *   - k: 64 - Fixed dimensionality
 * - Construction:
 *   - From [point, label] pairs, scaled by RobustScaler
 *   - Median splits, cycling dimensions 0-63, stops at 1 point/leaf
 * - Query:
 *   - k=3 NN with weighted Euclidean distance:
 *     sqrt(sum((x[i] - y[i])^2 for i=0..62) + w^2 * (x[63] - y[63])^2)
 *   - w (e.g., 10) amplifies handedness to punish differing hands
 *   - Majority vote on labels for ASL prediction
 * - Serialization:
 *   - Preorder: nodeType (uint8) [0=null, 1=node], point (64 * float32),
 *     label (uint8), splitDim (uint8), splitValue (float32), then left, right
 *   - Node size: 263 bytes (1 + 256 + 1 + 1 + 4)
 */

/**
 * Represents a node in the KD-tree for ASL detection.
 * Stores a 64D point (scaled hand landmarks + handedness), an ASCII label,
 * and split information for spatial partitioning.
 */
export class KDTreeNode {
	point: Float32Array
	label: number
	splitDim: number
	splitValue: number
	left: KDTreeNode | null
	right: KDTreeNode | null

	/**
	 * Creates a new KD-tree node.
	 * @param {Float32Array} point - The 64D scaled point (landmarks + handedness).
	 * @param {number} label - The uint8 ASCII label (0-255, e.g., 65 for "A").
	 * @param {number} splitDim - The dimension to split on (0-63).
	 * @param {number} splitValue - The split value from point[splitDim].
	 * @throws {Error} If point isn’t 64D, label isn’t uint8, or splitDim isn’t 0-63.
	 */
	constructor(point: Float32Array, label: number, splitDim: number, splitValue: number) {
		if (point.length !== 64) throw new Error(`Point must be 64D, got ${point.length}D`)
		if (label < 0 || label > 255) throw new Error(`Label must be uint8 (0-255), got ${label}`)
		if (splitDim < 0 || splitDim > 63) throw new Error(`splitDim must be 0-63, got ${splitDim}`)

		this.point = point
		this.label = label
		this.splitDim = splitDim
		this.splitValue = splitValue
		this.left = null
		this.right = null
	}
}

/** [point, label] pair for building the tree */
export type KDTreePoint = [Float32Array, number] // [point, label]

/**
 * A KD-tree for ASL detection, storing 64D scaled hand landmark vectors
 * with ASCII labels for k=3 nearest neighbor queries.
 */
export class KDTree {
	root: KDTreeNode | null
	readonly k: number = 64

	constructor() {
		this.root = null
	}

	build(data: KDTreePoint[]): void {
		if (!data || data.length === 0) throw new Error("Cannot build KD-tree from empty data")
		if (data.some(([point]) => point.length !== this.k)) throw new Error(`All points must be ${this.k}D`)
		this.root = this.buildRecursive(data, 0)
	}

	private buildRecursive(data: KDTreePoint[], depth: number): KDTreeNode | null {
		if (data.length === 0) return null
		if (data.length === 1) {
			const [point, label] = data[0]
			const splitDim = depth % this.k
			return new KDTreeNode(point, label, splitDim, point[splitDim])
		}
		const splitDim = depth % this.k
		data.sort((a, b) => a[0][splitDim] - b[0][splitDim])
		const medianIdx = Math.floor(data.length / 2)
		const [medianPoint, medianLabel] = data[medianIdx]
		const node = new KDTreeNode(medianPoint, medianLabel, splitDim, medianPoint[splitDim])
		const leftData = data.slice(0, medianIdx)
		const rightData = data.slice(medianIdx + 1)
		node.left = this.buildRecursive(leftData, depth + 1)
		node.right = this.buildRecursive(rightData, depth + 1)
		return node
	}

	private distance(p1: Float32Array, p2: Float32Array, handednessWeight: number): number {
		let sum = 0
		for (let i = 0; i < 63; i++) {
			const diff = p1[i] - p2[i]
			sum += diff * diff
		}
		const handednessDiff = p1[63] - p2[63]
		sum += (handednessWeight * handednessWeight) * (handednessDiff * handednessDiff)
		return Math.sqrt(sum)
	}

	query(query: Float32Array, k: number = 3, handednessWeight: number = 10): number {
		if (query.length !== this.k) throw new Error(`Query must be ${this.k}D`)
		if (!this.root) throw new Error("KD-tree is empty")

		const best = new SimpleMaxHeap<{ label: number; distance: number }>((a, b) => b.distance - a.distance)
		this.queryRecursive(this.root, query, k, handednessWeight, best)

		const neighbors = best.toArray()
		return this.majorityVote(neighbors)
	}

	private majorityVote(neighbors: Array<{ label: number; distance: number }>): number {
		if (neighbors.length === 0) throw new Error("No neighbors to vote on")

		// Map: label -> { count, totalDistance }
		const voteMap = new Map<number, { count: number; totalDistance: number }>()

		for (const { label, distance } of neighbors) {
			const entry = voteMap.get(label) || { count: 0, totalDistance: 0 }
			entry.count += 1
			entry.totalDistance += distance
			voteMap.set(label, entry)
		}

		// Find label with highest count, tiebreak by lowest total distance
		let maxCount = 0
		let minDistance = Infinity
		let winner = neighbors[0].label // Default to first neighbor if tied

		for (const [label, { count, totalDistance }] of voteMap) {
			if (count > maxCount || (count === maxCount && totalDistance < minDistance)) {
				maxCount = count
				minDistance = totalDistance
				winner = label
			}
		}

		return winner
	}

	private queryRecursive(
		node: KDTreeNode | null,
		query: Float32Array,
		k: number,
		handednessWeight: number,
		best: SimpleMaxHeap<{ label: number; distance: number }>,
	): void {
		if (!node) return

		const distSquared = this.distance(node.point, query, handednessWeight) ** 2
		const candidate = { label: node.label, distance: Math.sqrt(distSquared) }

		if (best.size() < k) {
			best.push(candidate)
		} else if (candidate.distance < best.peek()!.distance) {
			best.pop()
			best.push(candidate)
		}

		const splitDiff = query[node.splitDim] - node.splitValue
		const nearSide = splitDiff < 0 ? node.left : node.right
		const farSide = splitDiff < 0 ? node.right : node.left

		this.queryRecursive(nearSide, query, k, handednessWeight, best)

		const bestDist = best.size() === k ? best.peek()!.distance : Infinity
		if (best.size() < k || Math.abs(splitDiff) < bestDist) {
			this.queryRecursive(farSide, query, k, handednessWeight, best)
		}
	}

	/**
	 * Serializes the KD-tree to a binary ArrayBuffer using preorder traversal.
	 * Node format: [nodeType (uint8), point (64 * float32), label (uint8), splitDim (uint8), splitValue (float32)]
	 * Null: [0 (uint8)]. Total node size: 263 bytes.
	 * @returns {ArrayBuffer} Binary representation of the tree.
	 */
	serialize(): ArrayBuffer {
		// Calculate size: 263 bytes per node, 1 byte per null
		const nodeCount = this.countNodes(this.root)
		const bufferSize = nodeCount * 263 + (this.countNulls(this.root) * 1)
		const buffer = new ArrayBuffer(bufferSize)
		const view = new DataView(buffer)
		const offset = 0

		this.serializeRecursive(this.root, view, offset)
		return buffer
	}

	/**
	 * Counts the number of nodes in the tree.
	 * @param {KDTreeNode | null} node - Current node.
	 * @returns {number} Number of nodes.
	 */
	private countNodes(node: KDTreeNode | null): number {
		if (!node) return 0
		return 1 + this.countNodes(node.left) + this.countNodes(node.right)
	}

	/**
	 * Counts the number of null children in the tree.
	 * @param {KDTreeNode | null} node - Current node.
	 * @returns {number} Number of nulls.
	 */
	private countNulls(node: KDTreeNode | null): number {
		if (!node) return 1 // Null counts as 1
		return this.countNulls(node.left) + this.countNulls(node.right)
	}

	/**
	 * Recursively serializes the tree into the DataView.
	 * @param {KDTreeNode | null} node - Current node.
	 * @param {DataView} view - Binary view to write to.
	 * @param {number} offset - Current offset in bytes (passed by reference).
	 * @returns {number} Updated offset after writing.
	 */
	private serializeRecursive(node: KDTreeNode | null, view: DataView, offset: number): number {
		if (!node) {
			view.setUint8(offset, 0) // Null marker
			return offset + 1
		}

		// Node: 263 bytes
		view.setUint8(offset, 1) // Node marker
		offset += 1

		// Point: 64 float32 = 256 bytes
		for (let i = 0; i < 64; i++) {
			view.setFloat32(offset + i * 4, node.point[i], true) // Little-endian
		}
		offset += 256

		view.setUint8(offset, node.label) // 1 byte
		offset += 1

		view.setUint8(offset, node.splitDim) // 1 byte
		offset += 1

		view.setFloat32(offset, node.splitValue, true) // 4 bytes
		offset += 4

		// Recurse on children
		offset = this.serializeRecursive(node.left, view, offset)
		offset = this.serializeRecursive(node.right, view, offset)

		return offset
	}

	/**
	 * Deserializes a KD-tree from a binary ArrayBuffer.
	 * @param {ArrayBuffer} buffer - Binary data from serialize().
	 * @returns {KDTree} A new KD-tree instance.
	 */
	static deserialize(buffer: ArrayBuffer): KDTree {
		const tree = new KDTree()
		const view = new DataView(buffer)
		const offset = { value: 0 } // Object to pass by reference
		tree.root = tree.deserializeRecursive(view, offset)
		return tree
	}

	/**
	 * Recursively deserializes the tree from the DataView.
	 * @param {DataView} view - Binary view to read from.
	 * @param {{ value: number }} offset - Current offset (by reference).
	 * @returns {KDTreeNode | null} The root of this subtree.
	 */
	private deserializeRecursive(view: DataView, offset: { value: number }): KDTreeNode | null {
		const nodeType = view.getUint8(offset.value)
		offset.value += 1

		if (nodeType === 0) return null

		// Read point (64 float32 = 256 bytes)
		const point = new Float32Array(64)
		for (let i = 0; i < 64; i++) {
			point[i] = view.getFloat32(offset.value + i * 4, true)
		}
		offset.value += 256

		const label = view.getUint8(offset.value)
		offset.value += 1

		const splitDim = view.getUint8(offset.value)
		offset.value += 1

		const splitValue = view.getFloat32(offset.value, true)
		offset.value += 4

		const node = new KDTreeNode(point, label, splitDim, splitValue)
		node.left = this.deserializeRecursive(view, offset)
		node.right = this.deserializeRecursive(view, offset)

		return node
	}
}
