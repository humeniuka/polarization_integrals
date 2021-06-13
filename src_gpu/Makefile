#### build configuration ###########################################################
# You'll have to modify this section according to your GPU architecture and compiler

# compilers
NVCC := nvcc
NVCXX := gcc
## architecture and compute capability 
## GeForce GTX TITAN X
#CUARCH := -gencode=arch=compute_52,code=sm_52
#NVLINK := -arch=sm_52
## 1080ti
CUARCH := -gencode=arch=compute_61,code=sm_61
NVLINK := -arch=sm_61
# 
NVCXXFLAGS := -Wall -fPIC -O2 -DMKL_ILP64
NVCUFLAGS := -O2 -use_fast_math -std=c++11 -ccbin $(NVCXX) $(CUARCH) \
	-maxrregcount=63 --ptxas-options=-v --verbose --resource-usage \
	-Xptxas --disable-optimizer-constants
####################################################################################

# pybind11
INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

#
DEBUG_OPTS := --generate-line-info --source-in-ptx --verbose

all: lib
lib: lib_sp lib_dp

# Reminder about automatic variables in Makefiles:
#  $@  -  target of the rule
#  $^  -  all prerequisites
#  $<  -  first prerequisite

# python module with single precision integrals
lib_sp: export_sp.o polarization.o
	$(NVCC) $(NVLINK) -shared $^  -o ../polarization_integrals/_polarization_gpu_sp$(SUFFIX)

export_sp.o: export.cc polarization.h
	$(NVCC) --device-c --compiler-options "$(NVCXXFLAGS)"    \
          -DSINGLE_PRECISION                                     \
          $(NVCUFLAGS) $(INCLUDES)                               \
	  export.cc -o $@

# python module with double precision integrals
lib_dp: export_dp.o polarization.o
	$(NVCC) $(NVLINK) -shared $^  -o ../polarization_integrals/_polarization_gpu_dp$(SUFFIX)

export_dp.o: export.cc polarization.h
	$(NVCC) --device-c --compiler-options "$(NVCXXFLAGS)"     \
          $(NVCUFLAGS) $(INCLUDES)                                \
	  export.cc -o $@

# polarization integrals (both single and double precision functions)
polarization.o: polarization.cu Dawson_real.cu polarization.h
	@printf "Compiling %-25s > %-25s\n" $< $@
	$(NVCC) --device-c --compiler-options "$(NVCXXFLAGS)" \
          $(NVCUFLAGS) polarization.cu -o $@

Dawson_real.cu: Dawson.cu
	./double_to_real_cast.sh Dawson.cu Dawson_real.cu

#### standalone tests #####

tests: test_sp.x test_dp.x

# with single precision
test_sp.x: test_sp.o polarization.o
	$(NVCC) $(NVLINK) $^ -o $@

test_sp.o: test.cu polarization.h
	@printf "Compiling %-25s > %-25s\n" $< $@
	$(NVCC) --device-c --compiler-options "$(NVCXXFLAGS)" \
	  -DSINGLE_PRECISION                                  \
          $(NVCUFLAGS) test.cu -o $@

# with double precision
test_dp.x: test_dp.o polarization.o
	$(NVCC) $(NVLINK) $^ -o $@

test_dp.o: test.cu polarization.h
	@printf "Compiling %-25s > %-25s\n" $< $@
	$(NVCC) --device-c --compiler-options "$(NVCXXFLAGS)" \
          $(NVCUFLAGS) test.cu -o $@

# for inspecting the compiled code run
#   $ make polarization.ptx
%.ptx: %.cu
	$(NVCC) --compiler-options "$(NVCXXFLAGS)" \
	$(NVCUFLAGS) $(DEBUG_OPTS) -ptx $@

clean:
	rm -f *.o *.ptx *.x *~ *.nvprof \
		../polarization_integrals/_polarization_gpu$(SUFFIX) \
		../polarization_integrals/_polarization_gpu_sp$(SUFFIX)
