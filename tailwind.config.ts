import { type Config } from "tailwindcss"

export default {
	content: [
		"Web/{routes,islands,components}/**/*.{ts,tsx,js,jsx}",
	],
	theme: {
		extend: {
			animation: {
				flash: "flash 0.5s ease-in-out",
			},
			keyframes: {
				flash: {
					"0%": {
						filter: "brightness(1.2)",
					},
					"100%": {
						filter: "brightness(1)",
					},
				},
			},
		},
	},
} satisfies Config
