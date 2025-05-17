#!/bin/sh

# Check if the first argument is "serve"
if [ "$1" = "serve" ]; then
  # Shift the arguments to remove "serve"
  shift
  # Execute the uvicorn server with any remaining arguments (though typically none are needed here)
  exec uvicorn deployment.inference_server:app --host 0.0.0.0 --port 8080 "$@"
else
  # If the command is not "serve", execute the passed command
  exec "$@"
fi
