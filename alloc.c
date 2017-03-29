#include <stdlib.h>
#include <stdio.h>

#define ALIGNMENT 2*1024*1024

double * alloc(int len) {
  double * p = (double *)aligned_alloc(ALIGNMENT, sizeof(double)*len);
  printf("Allocated %p size %zu\n", p, sizeof(double)*len);
  return p;
}

