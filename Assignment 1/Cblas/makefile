CC=gcc
CFLAGS=-Wall -std=c99 -g
LDFLAGS=-L.
# uncomment below: Lib for the HPC server to make Cblas working.
# LDLIBS=-L /usr/lib64/atlas -lsatlas 
# uncomment below: Lib for Windows to make Cblas working.
LDLIBS=-lopenblas




ifeq ($(shell uname), Darwin)
	# Link against system default BLAS/LAPACK library on macOS
	LDLIBS=-llapack -lblas
endif

main: main_Tester.o

.PHONY: clean test
clean:
	-$(RM) *.o $(EXECUTABLE) main

test: main_Tester
	./$<