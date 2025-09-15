FROM node:18-slim

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --production

COPY . .

ENV REACT_APP_ORCH_API=http://orchestrator-service:3000

EXPOSE 3000

CMD ["npm", "start"]
