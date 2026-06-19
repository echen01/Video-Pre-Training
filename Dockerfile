# syntax=docker/dockerfile:1.7

FROM eclipse-temurin:8-jdk-jammy AS java8

FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

ARG VIRTUALGL_VERSION=3.1.4
ARG VIRTUALGL_SHA256=02edc6b599571c385389af1a006f07a70c298e1d97c580a9bfd4b39d835c51e6

ENV DEBIAN_FRONTEND=noninteractive \
    GRADLE_USER_HOME=/tmp/gradle \
    JAVA_HOME=/opt/java/openjdk \
    MALMO_MINECRAFT_OUTPUT_LOGDIR=/tmp/minerl/logs \
    MINERL_STATUS_DIR=/tmp/minerl/performance \
    MINERL_TMP_INSTANCES=1 \
    MINERL_WATCHERS_DIR=/tmp/minerl/watchers \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH="/opt/VirtualGL/bin:/app/.venv/bin:/opt/java/openjdk/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    VGL_DISPLAY=egl \
    VIRTUAL_ENV=/app/.venv \
    VPT_RENDER_BACKEND=auto \
    XVFB_WHD=1024x768x24

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        git \
        gzip \
        libasound2t64 \
        libcurl4t64 \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglu1-mesa \
        libglib2.0-0t64 \
        libglx-mesa0 \
        libnss3 \
        libsm6 \
        libuuid1 \
        libx11-6 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxss1 \
        libxt6 \
        libxtst6 \
        libxxf86vm1 \
        mesa-utils \
        patch \
        procps \
        tar \
        unzip \
        xauth \
        x11-utils \
        x11-xserver-utils \
        xvfb \
        xz-utils \
        zip \
        zstd && \
    curl -fsSL -o /tmp/virtualgl.deb \
        "https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
    echo "${VIRTUALGL_SHA256}  /tmp/virtualgl.deb" | sha256sum -c - && \
    apt-get install -y --no-install-recommends /tmp/virtualgl.deb && \
    rm -f /tmp/virtualgl.deb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=java8 /opt/java/openjdk /opt/java/openjdk

WORKDIR /app
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.gradle \
    MINERL_BUILD_MCP=1 uv sync --locked --no-editable && \
    chmod 755 /app/docker/vpt-minerl-entrypoint.sh && \
    mkdir -p /outputs /tmp/minerl /tmp/gradle && \
    chown -R nonroot:nonroot /app /outputs /tmp/minerl /tmp/gradle /home/nonroot && \
    chmod 1777 /outputs /tmp/minerl /tmp/gradle

USER nonroot
ENTRYPOINT ["/app/docker/vpt-minerl-entrypoint.sh"]
CMD ["python", "collect_policy.py", "--help"]
