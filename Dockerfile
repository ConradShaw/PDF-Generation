# Node 20 + Playwright Chromium for HTML→PDF workers
# The image tag must equal the "playwright" version in package.json (the browser build is version-specific).
FROM mcr.microsoft.com/playwright:v1.63.0-jammy

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --omit=dev

COPY tsconfig.json ./
COPY src ./src
COPY data ./data
COPY logo.png ./logo.png

ENV NODE_ENV=production
ENV PORT=8080
ENV RUN_WORKER=1

EXPOSE 8080

# Run node directly (not through npx/npm): the container's main process must be the app itself, or SIGTERM
# on a deploy kills the wrapper and the in-flight report job instead of letting the worker finish it.
CMD ["node", "--import", "tsx", "src/index.ts"]
