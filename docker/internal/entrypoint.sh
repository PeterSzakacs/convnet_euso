#!/bin/bash

if [ "$UID" -eq 0 ]; then
  >&2 echo "Error, please use a non-root user for launching this container"
  exit 1
else
  exec bash
fi
