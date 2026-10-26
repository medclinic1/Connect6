#!/bin/sh
# Lanzador para Linux/Mac: la GUI necesita un ejecutable, no un .py
cd "$(dirname "$0")" && exec python3 main.py
