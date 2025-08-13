CXX = icpx
# CXXFLAGS = -std=c++20 -O2 -fPIC -fsycl -fsycl-targets=spir64_gen
CXXFLAGS = -O3 -DNDEBUG -std=c++17 -g -fPIE -fsycl -fsycl-targets=intel_gpu_pvc -ftemplate-backtrace-limit=0
AOTFLAGS = --Xspirv-translator "-device pvc -options '-doubleGRF -vc-codegen -Xfinalizer -printregusage'"
CUTLASSFLAGS = -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier -fno-sycl-instrument-device-code

TORCH_DIR = /datadisk/wyz/miniforge-pypy3/envs/torch_val/lib/python3.10/site-packages/torch
LIB_DIR := $(TORCH_DIR)/lib

COMPILER_PATH = /opt/intel/oneapi/compiler/2025.1/include

# CXXOPTS_PATH = $(CURDIR)/cxxopts
INCLUDES = -I. -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(COMPILER_PATH)
LIBS = -L$(LIB_DIR) -ltorch -ltorch_xpu -ltorch_cpu -lc10 -lc10_xpu -Wl,-rpath,$(LIB_DIR)

CUTLASS_DIR = /datadisk/wyz/Workplace/cutlass-sycl
CUTLASS_INCLUDES = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/examples/common -I${CUTLASS_DIR}/tools/util/include -I$(CUTLASS_DIR)/applications/

SRC = paged_attn.cpp
OUT = test_paged_attn

all:
	$(CXX) $(CXXFLAGS) $(CUTLASSFLAGS) $(SRC) $(INCLUDES) $(CUTLASS_INCLUDES) $(LIBS) -o $(OUT)

clean:
	rm -f $(OUT)
