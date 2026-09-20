# Builds the status frontend and proxy backend into one image.

# Stage 1: Frontend Build
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python Backend Build
FROM python:3.11-slim AS backend-builder

WORKDIR /app

# Install uv, then the locked runtime deps into the system site-packages
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

# Export locked deps (no dev group, not the project itself) and pip-install them
RUN uv export --frozen --no-dev --no-emit-project > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend source and the vendored compat package the app imports at runtime
COPY backend/ ./backend/
COPY compat/ ./compat/

# Stage 3: Final Production Image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    supervisor \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy backend application and the vendored compat package
COPY --from=backend-builder /app/backend ./backend
COPY --from=backend-builder /app/compat ./compat
COPY --from=backend-builder /app/pyproject.toml ./

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Copy configuration files
COPY deployment/nginx/nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /etc/nginx/sites-enabled /etc/nginx/sites-available
COPY deployment/nginx/default.conf /etc/nginx/sites-enabled/default
COPY deployment/supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy startup script
COPY deployment/scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create necessary directories for nginx and fix permissions
RUN mkdir -p /var/cache/nginx /var/log/nginx /run && \
    touch /run/nginx.pid && \
    chown -R www-data:www-data /var/cache/nginx /var/log/nginx /run/nginx.pid && \
    chown -R www-data:www-data /usr/share/nginx/html

# Create app user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app/logs /app/data

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WORKERS=2

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start application
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
