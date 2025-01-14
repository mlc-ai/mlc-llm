#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

INPLACE_FORMAT=${INPLACE_FORMAT:=false}
LINT_ALL_FILES=true
REVISION=$(git rev-list --max-parents=0 HEAD)

while (($#)); do
    case "$1" in
    -i)
        INPLACE_FORMAT=true
        shift 1
        ;;
    --rev)
        LINT_ALL_FILES=false
        REVISION=$2
        shift 2
        ;;
    *)
        echo "Usage: clang-format.sh [-i] [--rev <commit>]"
        echo ""
        echo "Run clang-format on files that changed since <commit> or on all files in the repo"
        echo "Examples:"
        echo "- Compare last one commit: clang-format.sh --rev HEAD~1"
        echo "- Compare against upstream/main: clang-format.sh --rev upstream/main"
        echo "The -i will format files in-place instead of checking them."
        exit 1
        ;;
    esac
done

cleanup() {
    if [ -f /tmp/$$.clang-format.txt ]; then
        echo ""
        echo "---------clang-format log----------"
        cat /tmp/$$.clang-format.txt
    fi
    rm -rf /tmp/$$.clang-format.txt
}
trap cleanup 0

if [[ "$INPLACE_FORMAT" == "true" ]]; then
    echo "Running inplace git-clang-format against $REVISION"
    git-clang-format --extensions h,hh,hpp,c,cc,cpp,mm "$REVISION"
    exit 0
fi

if [[ "$LINT_ALL_FILES" == "true" ]]; then
    echo "Running git-clang-format against all C++ files"
    git-clang-format --diff --extensions h,hh,hpp,c,cc,cpp,mm "$REVISION" 1>/tmp/$$.clang-format.txt
else
    echo "Running git-clang-format against $REVISION"
    git-clang-format --diff --extensions h,hh,hpp,c,cc,cpp,mm "$REVISION" 1>/tmp/$$.clang-format.txt
fi

if grep --quiet -E "diff" </tmp/$$.clang-format.txt; then
    echo "clang-format lint error found. Consider running clang-format on these files to fix them."
    exit 1
fi
