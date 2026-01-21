FROM python:3.12.3-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN uv init --python python3.12

RUN uv add jinja2==3.1.6 fastapi==0.109.0 uvicorn==0.27.0 onnxruntime==1.23.2 pillow==10.2.0 numpy==1.26.3 python-multipart==0.0.6

COPY app/. .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host" , "0.0.0.0", "--port", "8000"]
