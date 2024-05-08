# Command to prepare the mlc llm static libraries
# This command will be invoked by prepare_package.sh in the subfolder
function help {
    echo -e "OPTION:"
    echo -e "  -s, --simulator                      Build for Simulator"
    echo -e "  -a, --arch        x86_64 | arm64     Simulator arch "
    echo -e "  -h,  --help                          Prints this help\n"
}

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

cmake ../..\
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
  -DUSE_METAL=ON

cmake --build . --config release --target mlc_llm_static -j
cmake --build . --target install --config release -j
cd ..

rm -rf MLCSwift/tvm_home
ln -s ../../3rdparty/tvm MLCSwift/tvm_home
