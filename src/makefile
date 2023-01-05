all:
	nvcc -I ../../cutlassBenchmark/cutlass/include/ benchmark/main.cu -o test
	./test
	rm test


build:
	nvcc -I ../../cutlassBenchmark/cutlass/include/ benchmark/main.cu -o test

run:
	./test

clean:
	rm test

tvm:
	python3 tvm/gemmGenerate.py 

tvm-rand:
	python3 tvm/gemmStochasticGenerate.py $M $N $K $t

dynamic:
	nvcc -I ../../cutlassBenchmark/cutlass/include/ benchmark/dynamic.cu -o test 
	./test $M $N $K
	rm test

debug:
	nvcc -I ../../cutlassBenchmark/cutlass/include/ benchmark/debug.cu -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_86,code=compute_86 -o test
	./test
	rm test

multistage:
	nvcc benchmark/multistage.cu -o test -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_86,code=compute_86
	./test
	rm test

gdb-build:
	nvcc -g -G -I ../../cutlassBenchmark/cutlass/include/ benchmark/debug.cu -o test
