export class SimpleMaxHeap<T> {
	private heap: T[] = []
	private compare: (a: T, b: T) => number

	constructor(compare: (a: T, b: T) => number) {
		this.compare = compare
	}

	push(item: T): void {
		this.heap.push(item)
		this.bubbleUp(this.heap.length - 1)
	}

	pop(): T | undefined {
		if (this.heap.length === 0) return undefined
		const root = this.heap[0]
		this.heap[0] = this.heap.pop()!
		this.bubbleDown(0)
		return root
	}

	peek(): T | undefined {
		return this.heap[0]
	}

	size(): number {
		return this.heap.length
	}

	toArray(): T[] {
		return [...this.heap]
	}

	private bubbleUp(index: number): void {
		while (index > 0) {
			const parent = Math.floor((index - 1) / 2)
			if (this.compare(this.heap[index], this.heap[parent]) <= 0) break
			;[this.heap[index], this.heap[parent]] = [this.heap[parent], this.heap[index]]
			index = parent
		}
	}

	private bubbleDown(index: number): void {
		const len = this.heap.length
		while (true) {
			let largest = index
			const left = 2 * index + 1
			const right = 2 * index + 2

			if (left < len && this.compare(this.heap[left], this.heap[largest]) > 0) largest = left
			if (right < len && this.compare(this.heap[right], this.heap[largest]) > 0) largest = right
			if (largest === index) break
			;[this.heap[index], this.heap[largest]] = [this.heap[largest], this.heap[index]]
			index = largest
		}
	}
}
