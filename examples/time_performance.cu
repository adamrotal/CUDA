#include <stdio.h>

int main(){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        cudaEventRecord(start);
        // Do Something Here
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Time elapsed: %f\n", milliseconds);
}
