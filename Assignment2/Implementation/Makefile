# Makefile

TARGET_J  = poisson_j		# Jacobi
TARGET_J_OMP  = poisson_j_omp		# JacobiOMP
TARGET_GS = poisson_gs		# Gauss-Seidel
TARGET_GS_OMP = poisson_gs_omp		# Gauss-SeidelOMP

SOURCES	= main.c print.c alloc3d.c jacobi.c jacobiOMP.c gauss_seidel.c gauss_seidelOMP.c sin_test.c init.c
OBJECTS	= print.o alloc3d.o init.o sin_test.o
MAIN_J	= main_j.o
MAIN_J_OMP	= main_j_omp.o
MAIN_GS = main_gs.o
MAIN_GS_OMP = main_gs_omp.o
OBJS_J	= $(MAIN_J) jacobi.o 
OBJS_J_OMP	= $(MAIN_J_OMP) jacobiOMP.o 
OBJS_GS	= $(MAIN_GS) gauss_seidel.o
OBJS_GS_OMP	= $(MAIN_GS_OMP) gauss_seidelOMP.o

# options and settings for the GCC compilers
#
CC		= gcc 
# DEFS	= -D_JACOBI
# DEFS	= -D_JACOBI_OMP
# DEFS	= -D_GAUSS_SEIDEL
# DEFS	= -D_GAUSS_SEIDEL_OMP 
OPT		= -g -std=c11 -fopenmp -Ofast
IPO		= 
SIN		= -D_SIN_TEST
ISA		=
CHIP	= 
ARCH	= 
PARA	= 
CFLAGS	= $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm 

all: $(TARGET_J) $(TARGET_GS) $(TARGET_J_OMP) $(TARGET_GS_OMP) 

$(TARGET_J): $(OBJECTS) $(OBJS_J)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J) $(OBJECTS) $(LDFLAGS)

$(TARGET_J_OMP): $(OBJECTS) $(OBJS_J_OMP)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J_OMP) $(OBJECTS) $(LDFLAGS)

$(TARGET_GS): $(OBJECTS) $(OBJS_GS)
	$(CC) -o $@ $(CFLAGS) $(OBJS_GS) $(OBJECTS) $(LDFLAGS)

$(TARGET_GS_OMP): $(OBJECTS) $(OBJS_GS_OMP)
	$(CC) -o $@ $(CFLAGS) $(OBJS_GS_OMP) $(OBJECTS) $(LDFLAGS)

$(MAIN_J):
	$(CC) -o $@ -D_JACOBI $(CFLAGS) -c main.c 

$(MAIN_J_OMP):
	$(CC) -o $@ -D_JACOBI_OMP $(CFLAGS) -c main.c 

$(MAIN_GS):
	$(CC) -o $@ -D_GAUSS_SEIDEL $(CFLAGS) -c main.c 

$(MAIN_GS_OMP):
	$(CC) -o $@ -D_GAUSS_SEIDEL_OMP $(CFLAGS) -c main.c 

clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET_J)  $(TARGET_GS)

# DO NOT DELETE

main_j.o: main.c print.h jacobi.h sin_test.h 
main_gs.o: main.c print.h gauss_seidel.h sin_test.h jacobi.h 
main_gs_omp.o: main.c print.h gauss_seidelOMP.h sin_test.h jacobi.h 
print.o: print.h
sin_test.o: sin_test.h jacobi.h
init.o: init.h 


