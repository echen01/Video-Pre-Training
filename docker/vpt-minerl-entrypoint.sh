#!/usr/bin/env bash
set -e

export GRADLE_USER_HOME="${GRADLE_USER_HOME:-/tmp/gradle}"
export MALMO_MINECRAFT_OUTPUT_LOGDIR="${MALMO_MINECRAFT_OUTPUT_LOGDIR:-/tmp/minerl/logs}"
export MINERL_STATUS_DIR="${MINERL_STATUS_DIR:-/tmp/minerl/performance}"
export MINERL_TMP_INSTANCES="${MINERL_TMP_INSTANCES:-1}"
export MINERL_WATCHERS_DIR="${MINERL_WATCHERS_DIR:-/tmp/minerl/watchers}"
export PATH="/opt/VirtualGL/bin:$PATH"

prepend_pytorch_cuda_libs() {
    local site_packages
    local cuda_libs=""

    for site_packages in /app/.venv/lib/python*/site-packages; do
        [ -d "$site_packages" ] || continue
        for libdir in "$site_packages"/nvidia/*/lib; do
            [ -d "$libdir" ] || continue
            cuda_libs="${cuda_libs:+$cuda_libs:}$libdir"
        done
    done

    if [ -n "$cuda_libs" ]; then
        export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

prepend_pytorch_cuda_libs

cleanup() {
    if [ -n "${VPT_XVFB_PID:-}" ] && kill -0 "$VPT_XVFB_PID" >/dev/null 2>&1; then
        kill "$VPT_XVFB_PID" >/dev/null 2>&1 || true
        wait "$VPT_XVFB_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p \
    "$GRADLE_USER_HOME" \
    "$MALMO_MINECRAFT_OUTPUT_LOGDIR" \
    "$MINERL_STATUS_DIR" \
    "$MINERL_WATCHERS_DIR"

case "${1:-}" in
    bash|/bin/bash|sh|/bin/sh)
        exec "$@"
        ;;
esac

has_gpu_device() {
    [ -e /dev/nvidiactl ] || [ -d /proc/driver/nvidia ] || [ -d /dev/dri ]
}

choose_backend() {
    case "${VPT_RENDER_BACKEND:-auto}" in
        auto)
            if command -v vglrun >/dev/null 2>&1 && has_gpu_device; then
                printf '%s\n' "virtualgl"
            elif [ -z "${DISPLAY:-}" ]; then
                printf '%s\n' "xvfb"
            else
                printf '%s\n' "native"
            fi
            ;;
        virtualgl|vgl|gpu)
            printf '%s\n' "virtualgl"
            ;;
        xvfb|software)
            printf '%s\n' "xvfb"
            ;;
        native|none)
            printf '%s\n' "native"
            ;;
        *)
            echo "Unknown VPT_RENDER_BACKEND=${VPT_RENDER_BACKEND}" >&2
            exit 2
            ;;
    esac
}

start_xvfb() {
    if [ -n "${DISPLAY:-}" ]; then
        return
    fi

    if [ -n "${VPT_DISPLAY:-}" ]; then
        export DISPLAY="$VPT_DISPLAY"
    else
        for candidate in $(seq 99 199); do
            if [ ! -S "/tmp/.X11-unix/X${candidate}" ] && [ ! -e "/tmp/.X${candidate}-lock" ]; then
                export DISPLAY=":${candidate}"
                break
            fi
        done
    fi
    if [ -z "${DISPLAY:-}" ]; then
        echo "Could not find a free X display in :99-:199" >&2
        exit 1
    fi

    local display_number="${DISPLAY#:}"
    display_number="${display_number%%.*}"
    Xvfb "$DISPLAY" \
        -ac \
        -screen 0 "${XVFB_WHD:-1024x768x24}" \
        -dpi 72 \
        +extension RANDR \
        +extension GLX \
        +iglx \
        +extension MIT-SHM \
        +render \
        -nolisten tcp \
        -noreset &

    VPT_XVFB_PID="$!"
    for _ in $(seq 1 50); do
        if [ -S "/tmp/.X11-unix/X${display_number}" ]; then
            return
        fi
        if ! kill -0 "$VPT_XVFB_PID" >/dev/null 2>&1; then
            wait "$VPT_XVFB_PID"
        fi
        sleep 0.1
    done

    echo "Timed out waiting for Xvfb display ${DISPLAY}" >&2
    exit 1
}

backend="$(choose_backend)"
case "$backend" in
    virtualgl)
        start_xvfb
        export VGL_DISPLAY="${VGL_DISPLAY:-egl}"
        if [ "${LIBGL_ALWAYS_SOFTWARE:-}" = "1" ]; then
            unset LIBGL_ALWAYS_SOFTWARE
        fi
        vglrun -d "$VGL_DISPLAY" "$@"
        exit $?
        ;;
    xvfb)
        export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
        if [ -z "${DISPLAY:-}" ]; then
            exec xvfb-run -a -s "-screen 0 ${XVFB_WHD:-1024x768x24}" "$@"
        fi
        exec "$@"
        ;;
esac

exec "$@"
