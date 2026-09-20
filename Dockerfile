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
FROM python:3.11.16-alpine3.24 AS backend-builder

WORKDIR /app

# Install uv only in the builder, then create an empty production virtual environment.
RUN pip install --no-cache-dir uv \
    && python -m venv --without-pip /opt/venv

COPY pyproject.toml uv.lock ./

# Export locked runtime dependencies and install them into the production virtual environment.
RUN uv export --frozen --no-dev --no-emit-project > requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-cache-dir -r requirements.txt

# Copy backend source and the vendored compat package the app imports at runtime
COPY backend/ ./backend/
COPY compat/ ./compat/

# Stage 3: Final Production Image
FROM python:3.11.16-alpine3.24

# Install nginx and remove Python build tools from the base image.
RUN apk add --no-cache nginx \
    && python -m pip uninstall --yes pip setuptools wheel

WORKDIR /app

# Copy only locked production Python dependencies and runtime commands.
COPY --from=backend-builder /opt/venv /opt/venv

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
    chown -R nginx:nginx /var/cache/nginx /var/log/nginx /run/nginx.pid && \
    chown -R nginx:nginx /usr/share/nginx/html

# Create app user
RUN adduser -D -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app/logs /app/data

# Environment variables
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WORKERS=1

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c 'from urllib.request import urlopen; urlopen("http://localhost:8080/api/health", timeout=5).close()'

# Start application
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
