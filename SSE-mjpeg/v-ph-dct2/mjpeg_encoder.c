#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include <xmmintrin.h> // SSE
#include <smmintrin.h> // SSE
#include <pmmintrin.h> // SSE

#include <sys/types.h>
#include <sys/sysctl.h>

#include "../v-orig/mjpeg_encoder.h"

static char *output_file;
static char *input_file;
static FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;
static uint32_t yph;
static uint32_t ypw;
static uint32_t uph;
static uint32_t upw;
static uint32_t vph;
static uint32_t vpw;

static uint32_t bit_buffer = 0;
static uint32_t bit_buffer_width = 0;

/* getopt */
extern int optind;
extern char *optarg;

static float dct_norm_table[8][8];
static void init_dct_norm_table( )
{
    int v;
    int u;
    for( v=0; v<8; v++ )
    {
        for( u=0; u<8; u++ )
        {
	  float a1 = !u ? ISQRT2 : 1.0f;
	  float a2 = !v ? ISQRT2 : 1.0f;
	  
	  dct_norm_table[v][u] =  a1*a2/4.0f;
	}
    }
}

static float costable[8][8];
static void init_costable( )
{
    int i;
    int u;
    for( i=0; i<8; i++ )
    {
        for( u=0; u<8; u++ )
        {
            costable[i][u] = cos((2*i+1)*u*PI/16.0f);
        }
    }
}



// Read RDTSC register - clock cycle counter
static __inline__ unsigned long long RDTSC(void)
{
#if defined(__i386__)
    unsigned long long int retval;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (retval));
    return retval;
#elif defined(__x86_64__)
    unsigned hi, lo;
    __asm__ volatile (".byte 0x0f, 0x31" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
#else
# error missing RDTSC code
#endif
}

// Read the CPU clock frequency to calculate times from cycles
uint64_t get_cpu_freq(void)
{
    uint64_t freq = 0;

#if defined (linux)    // On a Linux machine
    printf("On a LINUX - setting CPU freq manually\n");
    freq = 3400000000; // Manually read from the 2018 powermac.ifi.uio.no: cat /proc/cpuinfo
#else                  // Assuming a Mac - defined (__APPLE__)
    size_t size = sizeof(freq);	
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) < 0) { perror("sysctl"); }
#endif

    fprintf(stdout, "\nCPU FREQUENCY on this machine: %ld Hz\n\n", (long)freq);

    return freq;
}

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file)
{
    size_t len = 0;
    yuv_t *image = malloc(sizeof(yuv_t));


    /* Read Y' */
    image->Y = malloc(width*height);
    len += fread(image->Y, 1, width*height, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read U */
    image->U = malloc(width*height);
    len += fread(image->U, 1, (width*height)/4, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    /* Read V */
    image->V = malloc(width*height);
    len += fread(image->V, 1, (width*height)/4, file);
    if(ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if(len != width*height*1.5)
    {
        printf("Reached end of file.\n");
        return NULL;
    }

    return image;
}

static void dct_quantize(uint8_t *in_data, uint32_t width, uint32_t height,
        int16_t *out_data, uint32_t padwidth,
        uint32_t padheight, uint8_t *quantization)
{
    int y,x,u,v,j;

    /* Perform the DCT and quantization */
    for(y = 0; y < height; y += 8)
    {
        int jj = height - y;
        jj = MIN(jj, 8); // For the border-pixels, we might have a part of an 8x8 block

        for(x = 0; x < width; x += 8)
        {
            int ii = width - x;
            ii = MIN(ii, 8); // For the border-pixels, we might have a part of an 8x8 block

            //Loop through all elements of the block
            for(u = 0; u < 8; ++u)
            {
                for(v = 0; v < 8; ++v)
                {
                    /* Compute the DCT */
                    float dct = 0;
		    __m128 dct_vec = _mm_setzero_ps();

                    for(j = 0; j < jj; ++j) 
		    {  
		      __m128 coeff_v1, coeff_v2, cos_jv, cos_iu1, cos_iu2, const128;

		      const128 = _mm_set_ps1(128.0f);   // the 128.0f constant
		      int tmp = ((y+j)*width)+x;        // the static offset in the loop, only i varies

		      // We operate on 32-bit floats --> the 128 bit can hold 4
		      coeff_v1 = _mm_set_ps(in_data[tmp+0], // read in first 4 elements from in_data
					     in_data[tmp+1],
					     in_data[tmp+2],
					     in_data[tmp+3]);

		      coeff_v2 = _mm_set_ps(in_data[tmp+4], // read in next 4 elements from in_data
					     in_data[tmp+5],
					     in_data[tmp+6],
					     in_data[tmp+7]);

		      // Substract the 128.0f constant
		      coeff_v1 = _mm_sub_ps(coeff_v1, const128);
		      coeff_v2 = _mm_sub_ps(coeff_v2, const128);

		      // costable[j][v] - static for the inner loop
		      cos_jv = _mm_set_ps1(costable[j][v]);

		      // costable[i][u] - i varies from 0-7
		      cos_iu1 = _mm_set_ps(costable[0][u], 
					   costable[1][u], 
					   costable[2][u], 
					   costable[3][u]); 

		      cos_iu2 = _mm_set_ps(costable[4][u], 
					   costable[5][u], 
					   costable[6][u], 
					   costable[7][u]); 

		      // Multiply the cosine values - 2 x 4 times
		      cos_iu1 = _mm_mul_ps(cos_iu1, cos_jv);
		      cos_iu2 = _mm_mul_ps(cos_iu2, cos_jv);

		      // Multiply with the coeff values - 2 x 4 times
		      coeff_v1 = _mm_mul_ps(cos_iu1, coeff_v1);
		      coeff_v2 = _mm_mul_ps(cos_iu2, coeff_v2);

		      // Add the coeffs
		      // Add the 4 coeffs to the dct_vec - adding the 8x8 matrix row by 1/2-row first
		      // Add pairs of coeff-values
		      coeff_v1 = _mm_add_ps(coeff_v1, coeff_v2);
		      dct_vec  = _mm_add_ps(coeff_v1, dct_vec);
		    }

		    // dct_vec has summed all columns of the 8x8 block 
		    // --> horizontally add dct_vec for the final dct before quantization
		    float dct_tmp[4];

		    dct_vec = _mm_hadd_ps (dct_vec, dct_vec);
		    dct_vec = _mm_hadd_ps (dct_vec, dct_vec);  // All elements now contain the sum of the coeffs
		    
		    _mm_store_si128((void*)dct_tmp, (__m128i)dct_vec);
		    dct += dct_tmp[0]; // or any of the 4

		    // Normalizing 
		    dct *= dct_norm_table[u][v]; // Scale according to normalizing function 
	   
                    /* Quantize */
                    out_data[(y+v)*width+(x+u)] = (int16_t)(floor(0.5f + dct / (float)(quantization[v*8+u])));
                }
	    }
        }
    }
}

static void put_byte(int byte)
{
    int status = fputc(byte, outfile);

    if (status == EOF) {
        fprintf(stderr, "Error writing byte\n");
        exit(EXIT_FAILURE);
    }
}

static void put_bytes(const void* data, unsigned int len)
{
    int n = fwrite(data, 1, len, outfile);

    if(n != len) {
        fprintf(stderr, "Error writing bytes\n");
        exit(-1);
    }
}

static void write_SOI()
{
    put_byte(0xff);
    put_byte(0xd8);
}

static void write_DQT()
{
    int16_t size = 2 + (3 * 65);

    put_byte(0xff);
    put_byte(0xdb);

    put_byte(size >> 8);
    put_byte(size & 0xff);

    put_byte(0);
    put_bytes(yquanttbl, 64);
    put_byte(1);
    put_bytes(uquanttbl, 64);
    put_byte(2);
    put_bytes(vquanttbl, 64);
}

static void write_SOF0()
{
    int16_t size = 8 + 3 * COLOR_COMPONENTS;

    /* Header marker */
    put_byte(0xff);
    put_byte(0xc0);

    /* Size of header */
    put_byte(size >> 8);
    put_byte(size & 0xff);

    /* Precision */
    put_byte(8);

    /* Width and height */
    put_byte(height >> 8);
    put_byte(height & 0xff);
    put_byte(width >> 8);
    put_byte(width & 0xff);

    put_byte(COLOR_COMPONENTS);

    put_byte(1); /* Component id */
    put_byte(0x22); /* hor | ver sampling factor FIXME Y(2,2), U(1,1), V(1,1) */
    put_byte(0); /* Quant. tbl. id */

    put_byte(2); /* Component id */
    put_byte(0x11); /* hor | ver sampling factor */
    put_byte(1); /* Quant. tbl. id */

    put_byte(3); /* Component id */
    put_byte(0x11); /* hor | ver sampling factor */
    put_byte(2); /* Quant. tbl. id */
}

static void write_DHT_HTS(uint8_t id, uint8_t *numlength, uint8_t* data)
{
    /* Find out how many codes we are to write */
    int i;
    int n = 0;
    for(i = 0; i < 16; ++i)
        n += numlength[i];

    put_byte(id);
    put_bytes(numlength, 16);
    put_bytes(data, n);
}

static void write_DHT()
{
    int16_t size = 0x01A2; /* 2 + n*(17+mi); */

    /* Define Huffman Table marker */
    put_byte(0xff);
    put_byte(0xc4);

    /* Length of segment */
    put_byte(size >> 8);
    put_byte(size & 0xff);

    /* Write the four huffman table specifications */
    write_DHT_HTS(0x00, DCVLC_num_by_length[0], DCVLC_data[0]); /* DC table 0 */
    write_DHT_HTS(0x01, DCVLC_num_by_length[1], DCVLC_data[1]); /* DC table 1 */
    write_DHT_HTS(0x10, ACVLC_num_by_length[0], ACVLC_data[0]); /* AC table 0 */
    write_DHT_HTS(0x11, ACVLC_num_by_length[1], ACVLC_data[1]); /* AC table 1 */
}

static void write_SOS()
{
    int16_t size = 6 + 2 * COLOR_COMPONENTS;

    put_byte(0xff);
    put_byte(0xda);

    put_byte(size >> 8);
    put_byte(size & 0xff);

    put_byte(COLOR_COMPONENTS);

    put_byte(1); /* Component id */
    put_byte(0x00); /* DC | AC huff tbl */
    put_byte(2); /* Component id */
    put_byte(0x11); /* DC | AC huff tbl */
    put_byte(3); /* Component id */
    put_byte(0x11); /* DC | AC huff tbl */
    put_byte(0); /* ss, first AC */
    put_byte(63); /* se, last AC */
    put_byte(0); /* ah | al */
}

static void write_EOI()
{
    put_byte(0xff);
    put_byte(0xd9);
}

static inline uint8_t bit_width(int16_t i)
{
    return (uint8_t) (ceil((log(abs(i)+1))*ILOG2));
}

/**
 * Adds a bit to the bitBuffer. A call to Flush() is needed
 * in order to write any remainding bits in the buffer before
 * writing using another function.
 */
static void put_bits(int16_t bits, uint8_t n)
{
    assert(n <= 24  && "Error writing bit");

    if(n == 0)
        return;

    bit_buffer <<= n;
    bit_buffer |= bits & ((1 << n) - 1);
    bit_buffer_width += n;

    while(bit_buffer_width >= 8) {
        uint8_t c = (uint8_t)(bit_buffer >> (bit_buffer_width - 8));
        put_byte(c);
        if(c == 0xff)
            put_byte(0);

        bit_buffer_width -= 8;
    }
}


/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
static void flush()
{
    if(bit_buffer > 0) {
        uint8_t c = bit_buffer << (8 - bit_buffer_width);
        put_byte(c);
        if(c == 0xff)
            put_byte(0);
    }

    bit_buffer = 0;
    bit_buffer_width = 0;
}

static void write_block(int16_t *in_data, uint32_t width, uint32_t height,
        uint32_t uoffset, uint32_t voffset, int16_t *prev_DC,
        int32_t cc)
{
    uint32_t zigzag, i, j;

    static int16_t block[64];
    int32_t num_ac = 0;

    /* ZigZag */
    
    for(zigzag = 0; zigzag < 64; zigzag++)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];
        block[zigzag] = in_data[(voffset+v)*width+(uoffset+u)];
    }

    /* ZigZag */

    /* Calculate DC component, and write to stream */
    int16_t dc = block[0] - *prev_DC;
    *prev_DC = block[0];
    uint8_t size = bit_width(dc);
    put_bits(DCVLC[cc][size],DCVLC_Size[cc][size]);
    if(dc < 0)
        dc = dc - 1;

    put_bits(dc, size);

    /* find the last nonzero entry of the ac-coefficients */
    for(j = 64; j > 1 && !block[j-1]; j--)
        ;

    /* Put the nonzero ac-coefficients */
    for(i = 1; i < j; i++)
    {
        int16_t ac = block[i];
        if(ac == 0)
        {
            if(++num_ac == 16)
            {
                put_bits(ACVLC[cc][15][0], ACVLC_Size[cc][15][0]);
                num_ac = 0;
            }
        }
        else
        {
            uint8_t size = bit_width(ac);
            put_bits(ACVLC[cc][num_ac][size], ACVLC_Size[cc][num_ac][size]);

            if(ac < 0)
                --ac;
            put_bits(ac, size);
            num_ac = 0;
        }
    }

    /* Put end of block marker */
    if(j < 64)
        put_bits(ACVLC[cc][0][0], ACVLC_Size[cc][0][0]);
}

static void write_interleaved_data_MCU(int16_t *dct, uint32_t wi, uint32_t he,
        uint32_t h, uint32_t v, uint32_t x,
        uint32_t y, int16_t *prev_DC, int32_t cc)
{
    uint32_t i, j, ii, jj;
    for(j = y*v*8; j < (y+1)*v*8; j += 8)
    {
        jj = he-8;
        jj = MIN(j, jj);

        for(i = x*h*8; i < (x+1)*h*8; i += 8)
        {
            ii = wi-8;
            ii = MIN(i, ii);

            write_block(dct, wi, he, ii, jj, prev_DC, cc);
        }
    }
}

static void write_interleaved_data(dct_t *out)
{
    int16_t prev_DC[3] = {0, 0, 0};
    uint32_t u, v;

    /* Set up which huffman tables we want to use */
    int32_t yhtbl = 0;
    int32_t uhtbl = 1;
    int32_t vhtbl = 1;

    /* Find the number of MCU's for the intensity */
    uint32_t ublocks = (uint32_t) (ceil(ypw/(float)(8.0f*YX)));
    uint32_t vblocks = (uint32_t) (ceil(yph/(float)(8.0f*YY)));

    /* Write the MCU's interleaved */
    for(v = 0; v < vblocks; ++v)
    {
        for(u = 0; u < ublocks; ++u)
        {
            write_interleaved_data_MCU(out->Ydct, ypw, yph, YX, YY, u, v, &prev_DC[0], yhtbl);
            write_interleaved_data_MCU(out->Udct, upw, uph, UX, UY, u, v, &prev_DC[1], uhtbl);
            write_interleaved_data_MCU(out->Vdct, vpw, vph, VX, VY, u, v, &prev_DC[2], vhtbl);
        }
    }

    flush();
}

static void encode(yuv_t *image)
{
    dct_t *out = malloc(sizeof(dct_t));
    out->Ydct = malloc(yph*ypw*(sizeof(*out->Ydct)));
    out->Udct = malloc(uph*upw*(sizeof(*out->Udct)));
    out->Vdct = malloc(vph*vpw*(sizeof(*out->Vdct)));

    /* DCT and Quantization */

    dct_quantize(image->Y, width, height, out->Ydct, ypw, yph, yquanttbl);
    dct_quantize(image->U, (width*UX/YX), (height*UY/YY), out->Udct, upw, uph, uquanttbl);
    dct_quantize(image->V, (width*VX/YX), (height*VY/YY), out->Vdct, vpw, vph, vquanttbl);

    /* Write headers */

    /* Start Of Image */
    write_SOI();
    /* Define Quantization Table(s) */
    write_DQT();
    /* Start Of Frame 0(Baseline DCT) */
    write_SOF0();
    /* Define Huffman Tables(s) */
    write_DHT();
    /* Start of Scan */
    write_SOS();

    write_interleaved_data(out);

    /* End Of Image */
    write_EOI();

    free(out->Ydct);
    free(out->Udct);
    free(out->Vdct);
    free(out);
}

static void print_help()
{
    fprintf(stderr, "Usage: ./mjpeg_encoder [options] input_file\n");
    fprintf(stderr, "Commandline options:\n");
    fprintf(stderr, "  -h                             height of images to compress\n");
    fprintf(stderr, "  -w                             width of images to compress\n");
    fprintf(stderr, "  -o                             Output file (.mjpg)\n");
    fprintf(stderr, "  [-f]                           Limit number of frames to encode\n");
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    int c;
    yuv_t *image;

    unsigned long long int total_encoding;  // RDTSC
    unsigned long long int perf_counter;

    uint64_t cpu_freq = get_cpu_freq();

    total_encoding = 0 - RDTSC();

    if(argc == 1)
    {
        print_help();
        exit(EXIT_FAILURE);
    }

    while((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
    {
        switch(c)
        {
        case 'h':
            height = atoi(optarg);
            break;
        case 'w':
            width = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'f':
            limit_numframes = atoi(optarg);
            break;
        default:
            print_help();
            break;
        }
    }


    if(optind >= argc)
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }

    outfile = fopen(output_file, "wb");
    if(outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    /* Calculate the padded width and height */
    ypw = (uint32_t)(ceil(width/8.0f)*8);
    yph = (uint32_t)(ceil(height/8.0f)*8);
    upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
    uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
    vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
    vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

    input_file = argv[optind];

    if (limit_numframes)
        printf("Limited to %d frames.\n", limit_numframes);

    FILE *infile = fopen(input_file, "rb");

    if(infile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    init_costable();
    init_dct_norm_table();

    /* Encode input frames */
    int numframes = 0;;
    while(!feof(infile))
    {
        image = read_yuv(infile);

        if (!image) {
            break;
        }

        printf("Encoding frame %3d, ", numframes); 
	perf_counter = 0 - RDTSC();
        encode(image);
	perf_counter += RDTSC();
        printf("... done! (used %lld cycles)\n", perf_counter);

        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);

	++numframes;
        if (limit_numframes && numframes >= limit_numframes)
            break;
    }

    fclose(outfile);
    fclose(infile);

    total_encoding += RDTSC();

    printf("\n** Total number of cycles: %lld (time: %f ms) \n", 
	   total_encoding, (float) total_encoding/cpu_freq*1000);
    printf("** Per frame cycles: %lld (time %f ms)\n\n", 
	   total_encoding/limit_numframes, 
	   (float) total_encoding/limit_numframes/cpu_freq*1000);

    return EXIT_SUCCESS;
}
