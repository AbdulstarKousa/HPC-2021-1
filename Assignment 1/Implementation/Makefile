TARGET	= libmatmult.so
LIBSRCS	= functions.c
LDLIBS=-L /usr/lib64/atlas -lsatlas
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g -Ofast -funroll-loops -flto -std=c11
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