# DONE 1: -O2
# DONE 2: -O2 -funroll-loops 
# DONE 3: -O2 -flto
# 4: -O2 -funroll-loops -flto
# 5: -O3
# 6: -O3 -funroll-loops 
# 7: -O3 -flto
# 8: -O3 -funroll-loops -flto
# 9: -Ofast
# 10: -Ofast -funroll-loops 
# 11: -Ofast -flto
# 12: -Ofast -funroll-loops -flto

TARGET	= libmatmult.so
LIBSRCS	= functions.c
LDLIBS=-L /usr/lib64/atlas -lsatlas
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g -O2 -flto -std=c11
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