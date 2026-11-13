# Node 20 + Playwright Chromium for HTML→PDF workers
FROM mcr.microsoft.com/playwright:v1.51.0-jammy

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --omit=dev

COPY tsconfig.json ./
COPY src ./src
COPY logo.png ./logo.png

ENV NODE_ENV=production
ENV PORT=8080
ENV RUN_WORKER=1

EXPOSE 8080

CMD ["npx", "tsx", "src/index.ts"]
