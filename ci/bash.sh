#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ci/bash.sh <CONTAINER_NAME> -e key value -v key value [COMMAND]"
    exit -1
fi

DOCKER_IMAGE_NAME=("$1")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"
DOCKER_BINARY="docker"
DOCKER_ENV="-e ENV_USER_ID=$(id -u) -e ENV_GROUP_ID=$(id -g)"
DOCKER_VOLUMNS="-v ${WORKSPACE}:/workspace -v ${SCRIPT_DIR}:/docker"

shift 1
while [[ $# -gt 0 ]]; do
    cmd="$1"
    if [[ $cmd == "-e" ]]; then
        env_key=$2
        env_value=$3
        shift 3
        DOCKER_ENV="${DOCKER_ENV} -e ${env_key}=${env_value}"
    elif [[ $cmd == "-v" ]]; then
        volumn_key=$2
        volumn_value=$3
        shift 3
        DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${volumn_key}:${volumn_value}"
    elif [[ $cmd == "-j" ]]; then
        num_threads=$2
        shift 2
        DOCKER_ENV="${DOCKER_ENV} -e NUM_THREADS=${num_threads} --cpus ${num_threads}"
    else
        break
    fi
done

if [ "$#" -eq 0 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    COMMAND=("$@")
fi

if [[ -n ${MLC_CI_SETUP_DEPS:-} ]]; then
    DOCKER_ENV="${DOCKER_ENV} -e MLC_CI_SETUP_DEPS=${MLC_CI_SETUP_DEPS}"
fi

# Use nvidia-docker if the container is GPU.
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    DOCKER_ENV="${DOCKER_ENV} -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    if type nvidia-docker 1> /dev/null 2> /dev/null; then
        DOCKER_BINARY=nvidia-docker
    else
        DOCKER_BINARY=docker
        DOCKER_ENV="${DOCKER_ENV} --gpus all"
    fi

    # nvidia-docker treats Vulkan as a graphics API, so we need to
    # request passthrough of graphics APIs.  This could also be set in
    # the Dockerfile.
    DOCKER_ENV="${DOCKER_ENV} -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility"

    # vulkan comaptibility
    ICD_SEARCH_LOCATIONS=(
        # https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#icd-discovery-on-linux
        /usr/local/etc/vulkan/icd.d
        /usr/local/share/vulkan/icd.d
        /etc/vulkan/icd.d
        /usr/share/vulkan/icd.d
        /etc/glvnd/egl_vendor.d
        /usr/share/glvnd/egl_vendor.d
    )
    for filename in $(find "${ICD_SEARCH_LOCATIONS[@]}" -name "*nvidia*.json" 2> /dev/null); do
    DOCKER_VOLUMNS="${DOCKER_VOLUMNS} -v ${filename}:${filename}:ro"
    done
fi

# Print arguments.
echo "DOCKER_BINARY ${DOCKER_BINARY}"
echo "WORKSPACE: ${WORKSPACE}"
echo "IMAGE NAME: ${DOCKER_IMAGE_NAME}"
echo "ENV VARIABLES: ${DOCKER_ENV}"
echo "VOLUMES: ${DOCKER_VOLUMNS}"
echo "COMMANDS: '${COMMAND[@]}'"

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).

${DOCKER_BINARY} run --rm --pid=host \
    -w /workspace \
    ${DOCKER_VOLUMNS} \
    ${DOCKER_ENV} \
    ${DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME} \
    ${COMMAND[@]}
