FROM python:3.11-slim

WORKDIR /app

# Keep it simple: install only what's needed for the API container.
# Forge/SDXL runs outside this container on the host GPU machine.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV HOST=0.0.0.0
ENV PORT=${PORT:-8080}

EXPOSE ${PORT}
CMD ["sh","-c","python -m uvicorn main:app --host ${HOST} --port ${PORT} --limit-max-requests 1000 --timeout-keep-alive 30"]


