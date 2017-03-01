// Sorting reference, Odd-Even Algorithm using CUDA
__global__ void odd_even_sort_gpu_kernel_gmem(int * const data, const int num_elem) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid_idx;
  int offset = 0; // Start off with even, then odd
  int num_swaps;

// Calculation maximum index for a given block
// Last block it is number of elements minus one
// Other blocks to end of block minus one
  const int tid_idx_max = min( (((blockIdx.x+1) * (blockDim.x*2))-1), (num_elem-1) );
  do {
	// Reset number of swaps
    num_swaps = 0;
    // Work out index of data
    tid_idx =(tid * 2) + offset;
	// If no array or block overrun
    if (tid_idx < tid_idx_max)  {
	  // Read values into registers
	  const int d0 = data[tid_idx];
	  const int d1 = data[tid_idx+1];
	  // Compare registers
	  if ( d0 > d1 ) {
		// Swap values if needed
		data[tid_idx] = d1;
		data[tid_idx+1] = d0;

		// Keep track that we did a swap
        num_swaps++;
	  }
	}
	// Switch from even to off, or odd to even
	if (offset == 0)
	  offset = 1;
	else
	  offset = 0;
  } while (__syncthreads_count(num_swaps) != 0);
}
