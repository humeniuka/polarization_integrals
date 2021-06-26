INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all: lib

lib: export.cc polarization.cc polarization.h Faddeeva.cc Faddeeva.hh radial_overlap.cc radial_overlap.h
	@echo "Building python module"
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INCLUDES) export.cc polarization.cc Faddeeva.cc radial_overlap.cc -o ../polarization_integrals/_polarization$(SUFFIX)

clean:
	rm -f ../polarization_integrals/_polarization$(SUFFIX)
