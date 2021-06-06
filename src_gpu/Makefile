INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

# options for NVCC
NVCC_OPTS :=  -maxrregcount=63 --ptxas-options=-v -std=c++11

all: test.x lib

lib: lib_single lib_double

lib_double: export.cc polarization.cu Dawson_real.cu polarization.h
	@echo "//// Building python module (double precision integrals) ////"
	nvcc -O3 --compiler-options="-Wall -fPIC" -shared                    \
	     $(NVCC_OPTS) $(INCLUDES)                                        \
	     export.cc polarization.cu                                       \
	  -o ../polarization_integrals/_polarization_gpu$(SUFFIX)

lib_single: export.cc polarization.cu Dawson_real.cu polarization.h
	@echo "//// Building python module (single precision integrals) ////"
	nvcc -O3 --compiler-options="-Wall -fPIC" -shared                   \
             -DSINGLE_PRECISION                                             \
	     $(NVCC_OPTS) $(INCLUDES)                                       \
	     export.cc polarization.cu                                      \
	  -o ../polarization_integrals/_polarization_gpu_sp$(SUFFIX)


test.x: test.cu polarization.cu Dawson_real.cu polarization.h
	nvcc $(NVCC_OPTS) polarization.cu test.cu -o test.x

Dawson_real.cu: Dawson.cu
	./double_to_real_cast.sh Dawson.cu Dawson_real.cu

clean:
	rm -f test.x ../polarization_integrals/_polarization_gpu$(SUFFIX) \
	             ../polarization_integrals/_polarization_gpu_sp$(SUFFIX)
