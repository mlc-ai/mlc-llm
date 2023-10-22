#!/usr/bin/env bash

#
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh <CONTAINER_NAME> [--no-gpu] [COMMAND]"
    exit -1
fi

if [ "$1" == "--no-gpu" ]; then
    ENABLE_NV_DOCKER=0
    shift
else
    ENABLE_NV_DOCKER=1
fi

DOCKER_IMAGE_NAME=("$1")


if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    shift 1
    COMMAND=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

# Use nvidia-docker if the container is GPU.
if [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_ENV="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    CUDA_ENV=""
fi

# If this is an wheel test command then pass the env var to docker.
if [[ ! -z $WHEEL_TEST ]]; then
    WHEEL_TEST="-e WHEEL_TEST=${WHEEL_TEST}"
fi

if [[ "${DOCKER_IMAGE_NAME}" == *"cu"* ]]; then
    if [ "$ENABLE_NV_DOCKER" -eq 1 ]; then
        if ! type "nvidia-docker" 1> /dev/null 2> /dev/null
        then
            DOCKER_BINARY="docker"
            CUDA_ENV=" --gpus all "${CUDA_ENV}
        else
            DOCKER_BINARY="nvidia-docker"
        fi
    else
        DOCKER_BINARY="docker"
    fi
else
    DOCKER_BINARY="docker"
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).

${DOCKER_BINARY} run --rm --pid=host\
    -v ${WORKSPACE}:/workspace \
    -v ${SCRIPT_DIR}:/docker \
    -w /workspace \
    ${CUDA_ENV} \
    ${WHEEL_TEST} \
    ${DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME} \
    ${COMMAND[@]}
