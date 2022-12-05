######## SGX SDK Settings ########

SGX_SDK ?= /opt/intel/sgxsdk
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

# Don't support 32bit in this sample
SGX_COMMON_FLAGS := -m64
SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
TDX_LIBRARY_PATH := /usr/lib64
SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r

SGX_COMMON_FLAGS += -O2

SGX_COMMON_FLAGS += -DSGX_TRUSTED -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls
SGX_COMMON_CFLAGS := $(SGX_COMMON_FLAGS) -Wjump-misses-init -Wstrict-prototypes -Wunsuffixed-float-constants
SGX_COMMON_CXXFLAGS := $(SGX_COMMON_FLAGS) -Wnon-virtual-dtor -std=c++11


Cpp_Files := ../src/main/cpp/quote_verification.cpp

TDX_Cpp_Files := ../src/main/cpp/tdx_quote_generation.cpp

Include_Paths := -I. -I$(SGX_SDK)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux
C_Flags := -fPIC -Wno-attributes $(Include_Paths) -DNDEBUG -UEDEBUG -UDEBUG

Cpp_Flags := $(C_Flags)

Link_Flags := -L$(SGX_LIBRARY_PATH) -lsgx_dcap_quoteverify -lsgx_uae_service -lsgx_urts -lpthread -ldl

TDX_Link_Flags := -L$(TDX_LIBRARY_PATH) -ltdx_attest -lpthread -ldl

Cpp_Objects := $(Cpp_Files:.cpp=.o)

Cpp_Shared_Objects := target/classes/libquote_verification.so

TDX_Cpp_Objects := $(TDX_Cpp_Files:.cpp=.o)

TDX_Cpp_Shared_Objects := target/classes/libtdx_attest.so

######## Objects ########
.PHONY: all 
all: $(Cpp_Shared_Objects) $(TDX_Cpp_Shared_Objects)  

%.o: %.cpp 
	@$(CXX) $(SGX_COMMON_CXXFLAGS) $(Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(Cpp_Shared_Objects): $(Cpp_Objects)
	@$(CXX) -shared -o $@ $^ $(Link_Flags)
	@echo "LINK =>  $@"

$(TDX_Cpp_Shared_Objects): $(TDX_Cpp_Objects)
	@$(CXX) -shared -o $@ $^ $(TDX_Link_Flags)
	@echo "LINK =>  $@"

.PHONY: clean

clean:
	@rm -f $(Cpp_Shared_Objects)  $(Cpp_Objects) $(TDX_Cpp_Shared_Objects)  $(TDX_Cpp_Objects)
