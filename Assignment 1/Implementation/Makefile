# DONE 1: -O2
# DONE 2: -O2 -funroll-loops 
# DONE 3: -O2 -flto
# DONE 4: -O2 -funroll-loops -flto
# DONE 5: -O3
# DONE 6: -O3 -funroll-loops 
# DONE 7: -O3 -flto
# DONE 8: -O3 -funroll-loops -flto
# DONE 9: -Ofast
# DONE 10: -Ofast -funroll-loops 
# DONE 11: -Ofast -flto
# DONE 12: -Ofast -funroll-loops -flto
# 13: No extra optimizations

TARGET	= libmatmult.so
LIBSRCS	= functions.c
LDLIBS=-L /usr/lib64/atlas -lsatlas
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g -std=c11
PIC	= -fPIC

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(LDLIBS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

functions.o: functions.h