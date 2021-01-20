# Makefile

TARGET_J  = poisson_j		# Jacobi
TARGET_J_GPU1 = poisson_j_gpu1


SOURCES	= main.cu print.cu alloc3d.cu alloc3d_gpu.cu transfer3d_gpu.cu jacobi.cu jacobi_gpu.cu sin_test.cu init.cu
OBJECTS	= print.o alloc3d.o alloc3d_gpu.o transfer3d_gpu.o init.o sin_test.o
MAIN_J	= main_j.o
MAIN_J_GPU1	= main_j_gpu1.o

OBJS_J	= $(MAIN_J) jacobi.o 
OBJS_J_GPU1	= $(MAIN_J_GPU1) jacobi_gpu.o 


OPT	= -g -O3 
PIC	=
XOPTS = -Xptxas=-v
ARCH  = -arch=sm_80
OMP   = -fopenmp

CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PIC) $(OMP)" $(ARCH) $(XOPTS) 

CUDA_PATH ?= /appl/cuda/11.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc



SOFLAGS =
XLIBS	= 

$(TARGET_J): $(OBJECTS) $(OBJS_J)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS) 

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

#Jacobi GPU
$(TARGET_J_GPU1): $(OBJECTS) $(OBJS_J_GPU1)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS) 

	#$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

$(MAIN_J):
	$(CXX) -o $@ -D_JACOBI $(CXXFLAGS) -c main.cu

$(MAIN_J_GPU1):
	$(CXX) -o $@ $(CXXFLAGS) -c main.cu 
	# -D_JACOBI

clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET_J) $(TARGET_J_GPU1)


# DO NOT DELETE

main_j.o: main.cu print.h jacobi.h sin_test.h
main_j_gpu1.o: main.cu print.h jacobi_gpu.h sin_test.h

print.o: print.h
sin_test.o: sin_test.h
init.o: init.h 


