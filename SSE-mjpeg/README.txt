SEE and AVX vector examples for the MJPEG example.


Note, the programs are written on a _MAC_, but all except the line
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) < 0)
should also work on Linux. (used to calculate ms instead of CPU cycles.)


The different directories have different optimizations. All contain a Makefile

- make - compiling
- make clean - clean up
- make foreman - run the program on the foreman.yuv video file one directody down
- make play - display the resulting video


The outline.txt file contains the order of the tests we performed in class.
