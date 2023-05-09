# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

LOCAL_PATH := $(call my-dir)
MY_PATH := $(abspath $(LOCAL_PATH))
LIB_PATH := $(MY_PATH)/../../../../../build/lib
TVM_HOME := $(MY_PATH)/../../../../../build/tvm_home

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE := local_mlc_llm
LOCAL_SRC_FILES := $(LIB_PATH)/libmlc_llm.a
include $(PREBUILT_STATIC_LIBRARY)

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE := local_sentencepiece
LOCAL_SRC_FILES := $(LIB_PATH)/libsentencepiece.a
include $(PREBUILT_STATIC_LIBRARY)

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE := local_tokenizers_cpp
LOCAL_SRC_FILES := $(LIB_PATH)/libtokenizers_cpp.a
include $(PREBUILT_STATIC_LIBRARY)

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE := local_tvm_runtime
LOCAL_SRC_FILES := $(LIB_PATH)/libtvm_runtime.a
include $(PREBUILT_STATIC_LIBRARY)

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE := local_model_android
LOCAL_SRC_FILES := $(LIB_PATH)/libmodel_android.a
include $(PREBUILT_STATIC_LIBRARY)

#####################################################

include $(CLEAR_VARS)
LOCAL_MODULE = tvm4j_runtime_packed
LOCAL_SRC_FILES := org_apache_tvm_native_c_api.cc
LOCAL_LDFLAGS := -L$(SYSROOT)/usr/lib/ -llog -pthread -ldl -lm
LOCAL_C_INCLUDES := $(TVM_HOME)/include \
                    $(TVM_HOME)/3rdparty/dlpack/include \
                    $(TVM_HOME)/3rdparty/dmlc-core/include \
                    $(MY_PATH)
# LOCAL_C_FLAGS := -static

LOCAL_WHOLE_STATIC_LIBRARIES := local_mlc_llm local_tvm_runtime local_model_android
LOCAL_STATIC_LIBRARIES := local_sentencepiece local_tokenizers_cpp 
LOCAL_CPP_FEATURES += exceptions
LOCAL_ARM_MODE := arm

include $(BUILD_SHARED_LIBRARY)
