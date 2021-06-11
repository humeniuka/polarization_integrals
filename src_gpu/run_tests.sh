#!/bin/bash

make tests

# profile test applications (get timings of kernels)
nvprof -fo log_sp.nvprof ./test_async_sp.x
nvprof -fo log_dp.nvprof ./test_async_dp.x

