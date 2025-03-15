export function handler() {
	return new Response("", { status: 301, headers: { Location: "/" } })
}
