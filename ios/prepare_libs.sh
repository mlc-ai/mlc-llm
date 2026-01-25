# Command to prepare the mlc llm static libraries
# This command will be invoked by the "mlc_llm package" command
function help {
    echo -e "OPTION:"
    echo -e "  -s, --simulator                      Build for Simulator"
    echo -e "  -a, --arch        x86_64 | arm64     Simulator arch "
    echo -e "  -c, --catalyst                       Build for Mac Catalyst (arm64 only)"
    echo -e "      --deployment-target VERSION     Mac Catalyst deployment target (default: 18.0)"
    echo -e "  -h,  --help                          Prints this help\n"
}

MLC_LLM_SOURCE_DIR="${MLC_LLM_SOURCE_DIR:-..}"
is_simulator="false"
is_catalyst="false"
arch="arm64"
deployment_target="18.0"

# rustup is required to install iOS target stdlibs, and we need to make sure
# the rustup-managed cargo/rustc are used during cross compilation.
if [ -d "${HOME}/.cargo/bin" ]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi
if ! command -v rustup >/dev/null 2>&1; then
  echo "error: rustup is required to build iOS static libraries." >&2
  echo "Install rustup and retry, e.g.:" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
  exit 1
fi

# Args while-loop
while [ "$1" != "" ];
do
   case $1 in
   -s  | --simulator  )   is_simulator="true"
                          ;;
   -c  | --catalyst  )    is_catalyst="true"
                          ;;
   -a  | --arch  )        shift
                          arch=$1
                          ;;
   --deployment-target )  shift
                          deployment_target=$1
                          ;;
   -h   | --help )        help
                          exit
                          ;;
   *)
                          echo "$script: illegal option $1"
                          usage
                                          exit 1 # error
                          ;;
    esac
    shift
done

set -euxo pipefail

sysroot="iphoneos"
type="Release"
build_dir="build"

if [ "$is_catalyst" = "true" ]; then
  if [ "$is_simulator" = "true" ]; then
    echo "error: --simulator is not supported with --catalyst." >&2
    exit 1
  fi
  if [ "$arch" != "x86_64" ]; then
    arch="arm64"
  fi
  sysroot="macosx"
  build_dir="build-maccatalyst-$arch"
fi

if [ "$is_simulator" = "true" ]; then
  if [ "$arch" = "arm64" ]; then
    # iOS simulator on Apple processors
    rustup target add aarch64-apple-ios-sim
  else
    # iOS simulator on x86 processors
    rustup target add x86_64-apple-ios
  fi
  sysroot="iphonesimulator"
  type="Debug"
else
  # iOS devices
  rustup target add aarch64-apple-ios
  if [ "$is_catalyst" = "true" ]; then
    if [ "$arch" = "x86_64" ]; then
      rustup target add x86_64-apple-ios-macabi
    else
      rustup target add aarch64-apple-ios-macabi
    fi
  fi
fi

mkdir -p "$build_dir" && cd "$build_dir"

cmake_args=(
  -DCMAKE_BUILD_TYPE="$type"
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON
  -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON
  -DCMAKE_INSTALL_PREFIX=.
  -DCMAKE_CXX_FLAGS="-O3"
  -DMLC_LLM_INSTALL_STATIC_LIB=ON
  -DUSE_METAL=ON
  -DTVM_FFI_USE_LIBBACKTRACE=OFF
  -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF
)

if [ "$is_catalyst" = "true" ]; then
  toolchain="$MLC_LLM_SOURCE_DIR/3rdparty/tokenizers-cpp/sentencepiece/cmake/ios.toolchain.cmake"
  if [ "$arch" = "x86_64" ]; then
    platform="MAC_CATALYST"
  else
    platform="MAC_CATALYST_ARM64"
  fi
  cmake_args+=(
    -DCMAKE_TOOLCHAIN_FILE="$toolchain"
    -DPLATFORM="$platform"
    -DDEPLOYMENT_TARGET="$deployment_target"
    -DENABLE_BITCODE=OFF
  )
else
  cmake_args+=(
    -DCMAKE_SYSTEM_NAME=iOS
    -DCMAKE_SYSTEM_VERSION=14.0
    -DCMAKE_OSX_SYSROOT="$sysroot"
    -DCMAKE_OSX_ARCHITECTURES="$arch"
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
  )
fi

cmake "$MLC_LLM_SOURCE_DIR" "${cmake_args[@]}"


cmake --build . --config release --target mlc_llm_static -j
cmake --build . --target install --config release -j
cd ..

rm -rf $MLC_LLM_SOURCE_DIR/ios/MLCSwift/tvm_home
ln -s $MLC_LLM_SOURCE_DIR/3rdparty/tvm $MLC_LLM_SOURCE_DIR/ios/MLCSwift/tvm_home
