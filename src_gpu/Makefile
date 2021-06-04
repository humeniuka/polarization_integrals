INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

# options for NVCC
NVCC_OPTS :=  -maxrregcount=63 --ptxas-options=-v -std=c++11

all: test.x lib

lib: export.cc polarization.cu Dawson.cu polarization.h
	@echo "Building python module"
	nvcc -O3 --compiler-options="-Wall -fPIC" -shared $(NVCC_OPTS) $(INCLUDES) \
	     export.cc polarization.cu \
	  -o ../polarization_integrals/_polarization_gpu$(SUFFIX)

test.x: test.cu polarization.cu Dawson.cu
	nvcc $(NVCC_OPTS) polarization.cu test.cu -o test.x

clean:
	rm -f test.x ../polarization_integrals/_polarization_gpu$(SUFFIX)
