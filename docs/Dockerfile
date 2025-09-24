FROM python:3.13-slim AS builder

WORKDIR /app

# Optional: system packages needed by some deps or for graphviz rendering
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Sphinx and theme/extensions used by the docs (pinned to project versions)
RUN pip install --no-cache-dir \
    sphinx==8.2.3 \
    furo==2025.7.19 \
    myst-parser==4.0.1 \
    autodoc-pydantic==2.2.0 \
    sphinx-autodoc-typehints==3.2.0

# Copy the entire project so autodoc can import the package
COPY . .

# Install the package (build from pyproject via PEP 517)
RUN pip install --no-cache-dir .

# Build docs (with -M so output goes to build/html)
WORKDIR /app/docs
RUN sphinx-build -M html source build -n -W --keep-going


FROM python:3.13-slim

WORKDIR /srv/docs

COPY --from=builder /app/docs/build/html .

CMD ["python", "-m", "http.server", "9898", "--bind", "0.0.0.0"]

