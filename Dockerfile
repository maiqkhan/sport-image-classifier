FROM python:3.12.3-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
ENTRYPOINT ["uvicorn", "main:app", "--host" , "0.0.0.0", "--port", "8000"]
