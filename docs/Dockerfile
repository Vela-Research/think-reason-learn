FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir \
    sphinx \
    furo \
    myst-parser \
    autodoc-pydantic \
    sphinx-autodoc-typehints

COPY docs/ ./docs/
WORKDIR /app/docs
RUN sphinx-build -M html source build


FROM python:3.13-slim

WORKDIR /srv/docs

COPY --from=builder /app/docs/build/html .

CMD ["python", "-m", "http.server", "9898", "--bind", "0.0.0.0"]

