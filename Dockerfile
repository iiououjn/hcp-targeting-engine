FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install --no-cache-dir .
RUN useradd --create-home appuser && mkdir -p /app/outputs && chown -R appuser:appuser /app
USER appuser
# Local demonstration only; no HCP data is baked into this container.
ENTRYPOINT ["python", "-m", "hcp_engine.cli"]
CMD ["run", "--n-hcps", "800", "--out", "outputs"]
