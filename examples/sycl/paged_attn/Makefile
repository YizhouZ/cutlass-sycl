CXX = icpx
# CXXFLAGS = -std=c++20 -O2 -fPIC -fsycl -fsycl-targets=spir64_gen -fsycl-targets=intel_gpu_bmg_g21
CXXFLAGS = -O3 -DNDEBUG -std=c++17 -fPIC -fsycl -fsycl-targets=spir64_gen -ftemplate-backtrace-limit=0
AOTFLAGS = -Xsycl-target-backend=spir64_gen '-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread'
CUTLASSFLAGS = -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
# -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier -fno-sycl-instrument-device-code

BINDFLAGS = -DTORCH_EXTENSION_NAME=paged_attention

TORCH_DIR = /home/wyz/miniforge-pypy3/envs/torch_dev/lib/python3.10/site-packages/torch
LIB_DIR := $(TORCH_DIR)/lib

COMPILER_PATH = /opt/intel/oneapi/compiler/2025.1/include
PYTHON_PATH = $(shell python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

# CXXOPTS_PATH = $(CURDIR)/cxxopts
INCLUDES = -I. -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(COMPILER_PATH) -I${PYTHON_PATH}
LIBS = -L$(LIB_DIR) -ltorch -ltorch_python -ltorch_xpu -ltorch_cpu -lc10 -lc10_xpu -Wl,-rpath,$(LIB_DIR)

CUTLASS_DIR = /home/wyz/Workspace/cutlass-sycl
CUTLASS_INCLUDES = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/examples/common -I${CUTLASS_DIR}/tools/util/include -I$(CUTLASS_DIR)/applications/

SRC = paged_attention.cpp
OUT = paged_attention.so

all:
	$(CXX) $(CXXFLAGS) $(CUTLASSFLAGS) $(AOTFLAGS) $(BINDFLAGS) $(SRC) $(INCLUDES) $(CUTLASS_INCLUDES) $(LIBS) -shared -o $(OUT)

clean:
	rm -f $(OUT)
