# Command to prepare the mlc llm static libraries
# This command will be invoked by the "mlc_llm package" command
function help {
    echo -e "OPTION:"
    echo -e "  -s, --simulator                      Build for Simulator"
    echo -e "  -a, --arch        x86_64 | arm64     Simulator arch "
    echo -e "  -h,  --help                          Prints this help\n"
}

MLC_LLM_SOURCE_DIR="${MLC_LLM_SOURCE_DIR:-..}"
is_simulator="false"
arch="arm64"

# Args while-loop
while [ "$1" != "" ];
do
   case $1 in
   -s  | --simulator  )   is_simulator="true"
                          ;;
   -a  | --arch  )        shift
                          arch=$1
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
fi

mkdir -p build/ && cd build/

cmake $MLC_LLM_SOURCE_DIR\
  -DCMAKE_BUILD_TYPE=$type\
  -DCMAKE_SYSTEM_NAME=iOS\
  -DCMAKE_SYSTEM_VERSION=14.0\
  -DCMAKE_OSX_SYSROOT=$sysroot\
  -DCMAKE_OSX_ARCHITECTURES=$arch\
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0\
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON\
  -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON\
  -DCMAKE_INSTALL_PREFIX=.\
  -DCMAKE_CXX_FLAGS="-O3"\
  -DMLC_LLM_INSTALL_STATIC_LIB=ON\
  -DUSE_METAL=ON\
  -DTVM_FFI_USE_LIBBACKTRACE=OFF\
  -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF


cmake --build . --config release --target mlc_llm_static -j
cmake --build . --target install --config release -j
cd ..

rm -rf $MLC_LLM_SOURCE_DIR/ios/MLCSwift/tvm_home
ln -s $MLC_LLM_SOURCE_DIR/3rdparty/tvm $MLC_LLM_SOURCE_DIR/ios/MLCSwift/tvm_home
