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