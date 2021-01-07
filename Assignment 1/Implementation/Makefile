TARGET	= libmatmult.so
LIBSRCS	= functions.c
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g -O3 -std=c11
PIC	= -fPIC

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

functions.o: functions.h