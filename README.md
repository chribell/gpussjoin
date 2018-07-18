
# gpussjoin
Set Similarity joins in the GPGPU paradigm

This work is based on the filter-verifcation framework developed by Mann (http://ssjoin.dbresearch.uni-salzburg.at/).

We employ the GPGPU paradigm in order to accelerate exact set similarity join. The CPU is responsible for the candidate generation (filtering phase) while verification is delegated to the GPU.
Due to execution overlap between the CPU and GPU, on large datasets where billions of candiate pairs are verified, there is a 2.6x speedup over the sequential implementation.

## Dependencies

* Boost Library (program_options component for cli arguments)
* cuda-api-wrappers https://github.com/eyalroz/cuda-api-wrappers 

## Building 
```
./build (Release|Debug) SM_ARCH
./build Release 61 #builds an executable for Compute Capability 6.1
```

## Arguments & Execution
```
  --algorithm arg       algorithm to (allpairs, ppjoin, groupjoin)
  --threshold arg       jaccard threshold
  --input arg           file, each line a record
  --threads arg         Number of threads per block
  --devmemory arg       device memory to use (e.g. 512M, 4GB)
  --scenario arg        gpu scenario to execute (1, 2, 3)
```

```
./set_sim_join --alogirthm allpairs --threshold 0.9 --input ~/datasets/dblp.txt --devmemory 1G --scenario 3
```

The datasets and the preprocess scripts can be found at http://ssjoin.dbresearch.uni-salzburg.at/datasets.html
