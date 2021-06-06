#!/bin/bash
#
# When compiling a single precision version of a CUDA source file
# all floating point constants have to be cast to single precision.
# Otherwise arithmetic operations involving those constants will
# be performed in double precision slowing down the whole computation.
#
# Consider the following code:
#
#  real x = 0.5;
#  real t = exp(x + 1.0);
#
# `real` is a template parameter that can be float or double. 
# Even if x is of type float, the sum x + 1.0 is of type double. Since
# the function exp(...) is overloaded for different argument types, the
# compiler will select the double precision version. To fix this the
# constant 1.0 has to be cast to float.
#
#  real t = exp(x + ((real) 1.0));
#
# The regular expression below replaces all occurrences of double constants
# by `((real) constant)`.
#

if [ "$#" -ne 2 ]
then
    echo ""
    echo "Usage: $(basename $0)  input  output"
    echo ""
    echo "  wrap all floating point constants in the input source code"
    echo "  with the type cast ((real) ...) and write the transformed"
    echo "  source code to the output file."
    echo ""
    echo "  For example, the expression 'exp(t + 1.0)' whould be replaced by 'exp(t + ((real) 1.0))'."
    echo ""
    exit 1
fi

input=$1
output=$2

if [ ! -f "$input" ]
then
    echo "Input source code file $input does not exist!"
    exit 1
fi

sed "s/\([0-9]\+\.[0-9]*\([eE][+\-]\?[0-9]\+\)\?\)/((real) \1)/g" $input > tmp
mv tmp $output



