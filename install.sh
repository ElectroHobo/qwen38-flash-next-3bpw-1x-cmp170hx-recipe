#!/usr/bin/env bash
# install.sh — get the pinned engine: pulls the engine image pinned in
# start.sh by digest. It is Morrowmake's release image for their Qwen EXL3
# recipe, used unchanged; its EXL3 extension is unused by a W4A16 checkpoint.
#
# Already there? Exits 0 without touching it.
# Pull anyway:  FORCE_INSTALL=1 ./install.sh
#
# Equivalent to: ./start.sh install
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" install
