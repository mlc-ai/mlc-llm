APP_PLATFORM := android-24
APP_ABI := arm64-v8a
APP_STL := c++_static

APP_CPPFLAGS += -DTVM_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++17 -Oz -frtti