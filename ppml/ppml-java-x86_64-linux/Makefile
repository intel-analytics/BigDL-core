######## SGX SDK Settings ########

SGX_SDK ?= /opt/intel/sgxsdk
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

# Don't support 32bit in this sample
SGX_COMMON_FLAGS := -m64
SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r

SGX_COMMON_FLAGS += -O2

SGX_COMMON_FLAGS += -DSGX_TRUSTED -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls
SGX_COMMON_CFLAGS := $(SGX_COMMON_FLAGS) -Wjump-misses-init -Wstrict-prototypes -Wunsuffixed-float-constants
SGX_COMMON_CXXFLAGS := $(SGX_COMMON_FLAGS) -Wnon-virtual-dtor -std=c++11


Urts_Library_Name := sgx_urts

App_Cpp_Files := ../src/main/cpp/quote_verification.cpp

App_Include_Paths := -I. -I$(SGX_SDK)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux
App_C_Flags := -fPIC -Wno-attributes $(App_Include_Paths)

App_C_Flags += -DNDEBUG -UEDEBUG -UDEBUG

App_Cpp_Flags := $(App_C_Flags)

App_Link_Flags := -L$(SGX_LIBRARY_PATH) -lsgx_dcap_quoteverify -l$(Urts_Library_Name) -lpthread -ldl

App_Link_Flags += -lsgx_uae_service

App_Cpp_Objects := $(App_Cpp_Files:.cpp=.o)

App_Cpp_Shared_Objects := target/classes/libquote_verification.so


######## App Objects ########
.PHONY: all 
all: $(App_Cpp_Shared_Objects)  

%.o: %.cpp 
	@$(CXX) $(SGX_COMMON_CXXFLAGS) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(App_Cpp_Shared_Objects): $(App_Cpp_Objects)
	@$(CXX) -shared -o $@ $^ $(App_Link_Flags)
	@echo "LINK =>  $@"


.PHONY: clean

clean:
	@rm -f $(App_Cpp_Shared_Objects)  $(App_Cpp_Objects) 

