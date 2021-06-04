INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all: lib

lib: export.cc polarization.cc polarization.h Faddeeva.cc Faddeeva.hh
	@echo "Building python module"
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INCLUDES) export.cc polarization.cc Faddeeva.cc -o ../polarization_integrals/_polarization$(SUFFIX)

clean:
	rm -f ../polarization_integrals/_polarization$(SUFFIX)
