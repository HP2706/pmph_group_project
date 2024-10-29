#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "tests/test_transpose_ker.cu"
#include "tests/test_scan_inc.cu"

int main() {
    initHwd();
   
    test_verify_transpose<1024>(1000000);


}
