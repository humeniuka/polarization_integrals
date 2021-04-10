INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all: polarization.cc polarization.h Faddeeva.cc Faddeeva.hh
	@echo "Building python module"
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INCLUDES) polarization.cc Faddeeva.cc -o ../polarization_integrals/polarization$(SUFFIX)

clean:
	rm -f ../polarization_integrals/polarization$(SUFFIX)
