hpc_gpu:
	make -f makefile.hpc ACTIVE_GPU=1
	
hpc_omp:
	make -f makefile.hpc ACTIVE_OMP=1
	
summit_gpu:
	make -f makefile.summit ACTIVE_GPU=1
	
summit_omp:
	make -f makefile.summit ACTIVE_OMP=1
	
clean:
	rm -rf ./build/triton
