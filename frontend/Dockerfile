# syntax=docker/dockerfile:1

# --- Stage 1: build the static assets -------------------------------------
# Node is only needed to compile the React app into plain HTML/CSS/JS.
FROM node:22-alpine AS builder
WORKDIR /app

# Copy lockfiles first so Docker can cache "npm ci" when only source changes.
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Stage 2: serve with Nginx --------------------------------------------
# The final image ships only the compiled assets behind Nginx — no Node,
# no node_modules — which keeps it tiny and fast to start.
FROM nginx:1.27-alpine AS runner

# Replace the default site with our SPA + reverse-proxy config.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
