#!/bin/sh
# ---------------------------------------------------------------------------
# Run the requested command as the owner of the mounted data/results folders.
#
# Why: `docker run` starts as root, so every file the workshop writes would land
# in your checkout owned by root and you would need sudo to delete it. The
# bind-mounted folders carry your own uid, so we read it and drop to it before
# running anything. Nothing is installed as root and no password is asked for.
#
# Set DAY4_KEEP_ROOT=1 to opt out and run as root (e.g. to experiment inside
# the image).
# ---------------------------------------------------------------------------
set -e

# mlflow prints "No username set in the environment" when neither is present;
# there is no login session inside the image to supply one.
export USER="${USER:-day4}" LOGNAME="${LOGNAME:-day4}"

if [ "$(id -u)" = "0" ] && [ "${DAY4_KEEP_ROOT:-0}" != "1" ]; then
    target_uid=""
    target_gid=""
    for probe in /app/data /app/results /app/notebooks; do
        if [ -d "$probe" ]; then
            owner="$(stat -c '%u' "$probe" 2>/dev/null || echo 0)"
            group="$(stat -c '%g' "$probe" 2>/dev/null || echo 0)"
            if [ "$owner" != "0" ]; then
                target_uid="$owner"
                target_gid="$group"
                break
            fi
        fi
    done

    if [ -n "$target_uid" ]; then
        HOME=/tmp/day4-home
        MPLCONFIGDIR=/tmp/day4-mpl
        UV_CACHE_DIR=/tmp/day4-uv
        export HOME MPLCONFIGDIR UV_CACHE_DIR
        exec setpriv --reuid "$target_uid" --regid "$target_gid" --clear-groups "$@"
    fi
fi

exec "$@"
