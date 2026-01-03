#!/bin/bash

echo "Fetching latest code from origin/main..."
git fetch origin
git reset --hard origin/main

echo "Starting the app..."
python app.py  # or whatever your main entry point is trying to edit this foi just because