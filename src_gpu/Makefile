INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

# options for NVCC
NVCC_OPTS :=  -maxrregcount=63 --ptxas-options=-v -std=c++11 --use_fast_math --verbose 
DEBUG_OPTS := --generate-line-info --source-in-ptx --verbose

all: lib
lib: lib_sp lib_dp

# python module with single precision integrals
lib_sp: export_sp.o polarization.o
	nvcc -shared export_sp.o polarization.o                         \
	  -o ../polarization_integrals/_polarization_gpu_sp$(SUFFIX)

export_sp.o: export.cc polarization.h
	nvcc --device-c -O3 --compiler-options="-Wall -fPIC"      \
          -DSINGLE_PRECISION                                      \
          $(NVCC_OPTS) $(INCLUDES)                                \
	  export.cc -o export_sp.o

# python module with double precision integrals
lib_dp: export_dp.o polarization.o
	nvcc -shared export_dp.o polarization.o                         \
	  -o ../polarization_integrals/_polarization_gpu_dp$(SUFFIX)

export_dp.o: export.cc polarization.h
	nvcc --device-c -O3 --compiler-options="-Wall -fPIC"      \
          $(NVCC_OPTS) $(INCLUDES)                                \
	  export.cc -o export_dp.o

# polarization integrals (both single and double precision functions)
polarization.o: polarization.cu Dawson_real.cu polarization.h
	nvcc --device-c -O3 --compiler-options="-Wall -fPIC"      \
          $(NVCC_OPTS) polarization.cu -o polarization.o

Dawson_real.cu: Dawson.cu
	./double_to_real_cast.sh Dawson.cu Dawson_real.cu

#### standalone tests #####

tests: test_sp.x test_dp.x

# with single precision
test_sp.x: test_sp.o polarization.o
	nvcc test_sp.o polarization.o -o test_sp.x

test_sp.o: test.cu polarization.h
	nvcc --device-c -O3 --compiler-options="-Wall" \
	  -DSINGLE_PRECISION                           \
          $(NVCC_OPTS) test.cu -o test_sp.o

# with double precision
test_dp.x: test_dp.o polarization.o
	nvcc test_dp.o polarization.o -o test_dp.x

test_dp.o: test.cu polarization.h
	nvcc --device-c -O3 --compiler-options="-Wall" \
          $(NVCC_OPTS) test.cu -o test_dp.o

# for inspecting the compiled code run
#   $ make polarization.ptx
%.ptx: %.cu
	nvcc -O3 --compiler-options="-Wall" \
	$(NVCC_OPTS) $(DEBUG_OPTS) -ptx $<


clean:
	rm -f *.o *.ptx *.x \
		../polarization_integrals/_polarization_gpu$(SUFFIX) \
		../polarization_integrals/_polarization_gpu_sp$(SUFFIX)
