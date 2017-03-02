Compile cudasort.cu:
    nvcc cudasort.cu -o sort -gencode=arch=compute_20,code=\"sm_20,compute_20\"