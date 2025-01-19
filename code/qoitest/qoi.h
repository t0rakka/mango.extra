/*

QOI - The "Quite OK Image" format for fast, lossless image compression

Dominic Szablewski - https://phoboslab.org


-- LICENSE: The MIT License(MIT)

Copyright(c) 2021 Dominic Szablewski

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


-- About

QOI encodes and decodes images in a lossless format. An encoded QOI image is
usually around 10--30% larger than a decently optimized PNG image.

QOI outperforms simpler PNG encoders in compression ratio and performance. QOI
images are typically 20% smaller than PNGs written with stbi_image. Encoding is 
25-50x faster and decoding is 3-4x faster than stbi_image or libpng.


-- Synopsis

// Define `QOI_IMPLEMENTATION` in *one* C/C++ file before including this
// library to create the implementation.

#define QOI_IMPLEMENTATION
#include "qoi.h"

// Encode and store an RGBA buffer to the file system. The qoi_desc describes
// the input pixel data.
qoi_write("image_new.qoi", rgba_pixels, &(qoi_desc){
    .width = 1920,
    .height = 1080, 
    .channels = 4,
    .colorspace = QOI_SRGB
});

// Load and decode a QOI image from the file system into a 32bbp RGBA buffer.
// The qoi_desc struct will be filled with the width, height, number of channels
// and colorspace read from the file header.
qoi_desc desc;
void *rgba_pixels = qoi_read("image.qoi", &desc, 4);



-- Documentation

This library provides the following functions;
- qoi_read    -- read and decode a QOI file
- qoi_decode  -- decode the raw bytes of a QOI image from memory
- qoi_write   -- encode and write a QOI file
- qoi_encode  -- encode an rgba buffer into a QOI image in memory

See the function declaration below for the signature and more information.

If you don't want/need the qoi_read and qoi_write functions, you can define
QOI_NO_STDIO before including this library.

This library uses malloc() and free(). To supply your own malloc implementation
you can define QOI_MALLOC and QOI_FREE before including this library.


-- Data Format

A QOI file has a 14 byte header, followed by any number of data "chunks".

struct qoi_header_t {
    char     magic[4];   // magic bytes "qoif"
    uint32_t width;      // image width in pixels (BE)
    uint32_t height;     // image height in pixels (BE)
    uint8_t  channels;   // must be 3 (RGB) or 4 (RGBA)
    uint8_t  colorspace; // a bitmap 0000rgba where
                         //   - a zero bit indicates sRGBA, 
                         //   - a one bit indicates linear (user interpreted)
                         //   colorspace for each channel
};

The decoder and encoder start with {r: 0, g: 0, b: 0, a: 255} as the previous
pixel value. Pixels are either encoded as
 - a run of the previous pixel
 - an index into a previously seen pixel
 - a difference to the previous pixel value in r,g,b,a
 - full r,g,b,a values

A running array[64] of previously seen pixel values is maintained by the encoder
and decoder. Each pixel that is seen by the encoder and decoder is put into this
array at the position (r^g^b^a) % 64. In the encoder, if the pixel value at this
index matches the current pixel, this index position is written to the stream.

Each chunk starts with a 2, 3 or 4 bit tag, followed by a number of data bits. 
The bit length of chunks is divisible by 8 - i.e. all chunks are byte aligned.
All values encoded in these data bits have the most significant bit (MSB) on the
left.

The possible chunks are:

 - QOI_INDEX -------------
|         Byte[0]         |
|  7  6  5  4  3  2  1  0 |
|-------+-----------------|
|  0  0 |     index       |

2-bit tag b00
6-bit index into the color index array: 0..63


 - QOI_RUN_8 -------------
|         Byte[0]         |
|  7  6  5  4  3  2  1  0 |
|----------+--------------|
|  0  1  0 |     run      |

3-bit tag b010
5-bit run-length repeating the previous pixel: 1..32


 - QOI_RUN_16 --------------------------------------
|         Byte[0]         |         Byte[1]         |
|  7  6  5  4  3  2  1  0 |  7  6  5  4  3  2  1  0 |
|----------+----------------------------------------|
|  0  1  1 |                 run                    |

3-bit tag b011
13-bit run-length repeating the previous pixel: 33..8224


 - QOI_DIFF_8 ------------
|         Byte[0]         |
|  7  6  5  4  3  2  1  0 |
|-------+-----+-----+-----|
|  1  0 |  dr |  db |  bg |

2-bit tag b10
2-bit   red channel difference from the previous pixel between -2..1
2-bit green channel difference from the previous pixel between -2..1
2-bit  blue channel difference from the previous pixel between -2..1

The difference to the current channel values are using a wraparound operation, 
so "1 - 2" will result in 255, while "255 + 1" will result in 0.


 - QOI_DIFF_16 -------------------------------------
|         Byte[0]         |         Byte[1]         |
|  7  6  5  4  3  2  1  0 |  7  6  5  4  3  2  1  0 |
|----------+--------------|------------ +-----------|
|  1  1  0 |   red diff   |  green diff | blue diff |

3-bit tag b110
5-bit   red channel difference from the previous pixel between -16..15
4-bit green channel difference from the previous pixel between -8..7
4-bit  blue channel difference from the previous pixel between -8..7

The difference to the current channel values are using a wraparound operation, 
so "10 - 13" will result in 253, while "250 + 7" will result in 1.


 - QOI_DIFF_24 ---------------------------------------------------------------
|         Byte[0]         |         Byte[1]         |         Byte[2]         |
|  7  6  5  4  3  2  1  0 |  7  6  5  4  3  2  1  0 |  7  6  5  4  3  2  1  0 |
|-------------+----------------+--------------+----------------+--------------|
|  1  1  1  0 |   red diff     |   green diff |    blue diff   |  alpha diff  |

4-bit tag b1110
5-bit   red channel difference from the previous pixel between -16..15
5-bit green channel difference from the previous pixel between -16..15
5-bit  blue channel difference from the previous pixel between -16..15
5-bit alpha channel difference from the previous pixel between -16..15

The difference to the current channel values are using a wraparound operation, 
so "10 - 13" will result in 253, while "250 + 7" will result in 1.


 - QOI_COLOR -------------
|         Byte[0]         |
|  7  6  5  4  3  2  1  0 |
|-------------+--+--+--+--|
|  1  1  1  1 |hr|hg|hb|ha|

4-bit tag b1111
1-bit   red byte follows
1-bit green byte follows
1-bit  blue byte follows
1-bit alpha byte follows

For each set bit hr, hg, hb and ha another byte follows in this order. If such a
byte follows, it will replace the current color channel value with the value of
this byte.


The byte stream is padded at the end with 4 zero bytes. Size the longest chunk
we can encounter is 5 bytes (QOI_COLOR with RGBA set), with this padding we just
have to check for an overrun once per decode loop iteration.

*/


// -----------------------------------------------------------------------------
// Header - Public functions

#ifndef QOI_H
#define QOI_H

#ifdef __cplusplus
extern "C" {
#endif


// Encode raw RGB or RGBA pixels into a QOI image in memory. w and h denote the
// width and height of the pixel data. channels must be either 3 for RGB data 
// or 4 for RGBA.
// The function either returns NULL on failure (invalid parameters or malloc 
// failed) or a pointer to the encoded data on success. On success the out_len
// is set to the size in bytes of the encoded data.
// The returned qoi data should be free()d after user.

mango::u8* qoi_encode(const mango::u8* image, size_t stride, int w, int h, size_t* out_len);


// Decode a QOI image from memory into either raw RGB (channels=3) or RGBA 
// (channels=4) pixel data.
// The function either returns NULL on failure (invalid parameters or malloc 
// failed) or a pointer to the decoded pixels. On success out_w and out_h will
// be set to the width and height of the decoded image.
// The returned pixel data should be free()d after use.

void qoi_decode(unsigned char* image, const mango::u8* data, size_t size, int w, int h, size_t stride);

#ifdef __cplusplus
}
#endif
#endif // QOI_H


// -----------------------------------------------------------------------------
// Implementation

#ifdef QOI_IMPLEMENTATION

#define QOI_INDEX   0x00 // 00xxxxxx
#define QOI_RUN_8   0x40 // 010xxxxx
#define QOI_RUN_16  0x60 // 011xxxxx
#define QOI_DIFF_8  0x80 // 10xxxxxx
#define QOI_DIFF_16 0xc0 // 110xxxxx
#define QOI_DIFF_24 0xe0 // 1110xxxx
#define QOI_COLOR   0xf0 // 1111xxxx

#define QOI_UPDATE  0x80 // 1xxxxxxx

#define QOI_MASK_2  0xc0 // 11000000
#define QOI_MASK_3  0xe0 // 11100000
#define QOI_MASK_4  0xf0 // 11110000

//#define QOI_COLOR_HASH(C) (0x92458355 * C + 0xcb533df9)
#define QOI_COLOR_HASH(C) (C.r ^ C.g ^ C.b ^ C.a)
#define QOI_PADDING 4

using Color = mango::image::Color;
using u8 = mango::u8;
using u32 = mango::u32;

static inline
bool is_diff(int r, int g, int b, int a)
{
    return r > -16 && r < 17 &&
           g > -16 && g < 17 && 
           b > -16 && b < 17 && 
           a > -16 && a < 17;
}

static inline
bool is_diff8(int r, int g, int b, int a)
{
    return a == 0 &&
           r > -2 && r < 3 &&
           g > -2 && g < 3 && 
           b > -2 && b < 3;
}

static inline
bool is_diff16(int r, int g, int b, int a)
{
    return a == 0 &&
           r > -16 && r < 17 && 
           g >  -8 && g <  9 && 
           b >  -8 && b <  9;
}

u8* qoi_encode(const u8* image, size_t stride, int width, int height, size_t* out_len)
{
    if (image == NULL || out_len == NULL ||
        width <= 0 || width >= (1 << 16) ||
        height <= 0 || height >= (1 << 16))
    {
        return nullptr;
    }

    constexpr int channels = 4;

    int max_size = width * height * (channels + 1) + QOI_PADDING;
    u8* bytes = new u8[max_size];
    if (!bytes)
    {
        return nullptr;
    }

    int p = 0;

    Color index[64] = { 0 };

    int run = 0;
    Color prev(0, 0, 0, 255);
    Color color = prev;

    for (int y = 0; y < height; ++y)
    {
        bool is_last_scanline = (y == height - 1);
        int xend = is_last_scanline ? (width - 1) : (1 << 30);

        const Color* src = reinterpret_cast<const Color*>(image);

        for (int x = 0; x < width; ++x)
        {
            color = src[x];

            if (color == prev)
            {
                run++;
            }

            bool is_last_pixel = (x == xend);

            if (run > 0 && (run == 0x2020 || color != prev || is_last_pixel))
            {
                if (run < 33)
                {
                    run -= 1;
                    bytes[p++] = QOI_RUN_8 | run;
                }
                else
                {
                    run -= 33;
                    bytes[p++] = QOI_RUN_16 | run >> 8;
                    bytes[p++] = run;
                }
                run = 0;
            }

            if (color != prev)
            {
                int index_pos = QOI_COLOR_HASH(color) % 64;

                if (index[index_pos] == color)
                {
                    bytes[p++] = QOI_INDEX | index_pos;
                }
                else
                {
                    index[index_pos] = color;

                    int r = color.r - prev.r;
                    int g = color.g - prev.g;
                    int b = color.b - prev.b;
                    int a = color.a - prev.a;

                    if (is_diff(r, g, b, a))
                    {
                        if (is_diff8(r, g, b, a))
                        {
                            bytes[p++] = QOI_DIFF_8 | ((r + 1) << 4) | (g + 1) << 2 | (b + 1);
                        }
                        else if (is_diff16(r, g, b, a))
                        {
                            bytes[p++] = QOI_DIFF_16 | (r + 15);
                            bytes[p++] = ((g + 7) << 4) | (b + 7);
                        }
                        else
                        {
                            bytes[p++] = QOI_DIFF_24 | ((r + 15) >> 1);
                            bytes[p++] = ((r + 15) << 7) | ((g + 15) << 2) | ((b + 15) >> 3);
                            bytes[p++] = ((b + 15) << 5) | (a + 15);
                        }
                    }
                    else
                    {
                        int p0 = p;
                        bytes[p++] = QOI_COLOR;

                        int mask = 0;
                        if (r)
                        {
                            mask |= 8;
                            bytes[p++] = color.r;
                        }
                        if (g)
                        {
                            mask |= 4;
                            bytes[p++] = color.g;
                        }
                        if (b)
                        {
                            mask |= 2;
                            bytes[p++] = color.b;
                        }
                        if (a)
                        {
                            mask |= 1;
                            bytes[p++] = color.a;
                        }

                        bytes[p0] |= mask;
                    }
                }
            }

            prev = color;
        }

        image += stride;
    }

    for (int i = 0; i < QOI_PADDING; i++)
    {
        bytes[p++] = 0;
    }

    *out_len = p;
    return bytes;
}

/*

u8* qoi_encode(const u8* image, size_t stride, int width, int height, size_t* out_len)
{
    if (image == NULL || out_len == NULL ||
        width <= 0 || width >= (1 << 16) ||
        height <= 0 || height >= (1 << 16))
    {
        return nullptr;
    }

    constexpr int channels = 4;

    int max_size = width * height * (channels + 1) + QOI_PADDING;
    u8* bytes = new u8[max_size];
    if (!bytes)
    {
        return nullptr;
    }

    int p = 0;

    Color index[64] = { 0 };

    int run = 0;
    Color prev(0, 0, 0, 255);
    Color color = prev;

    for (int y = 0; y < height; ++y)
    {
        bool is_last_scanline = (y == height - 1);
        int xend = is_last_scanline ? (width - 1) : (1 << 30);

        const Color* src = reinterpret_cast<const Color*>(image);

        for (int x = 0; x < width; ++x)
        {
            color = src[x];

            if (color == prev)
            {
                run++;
            }

            bool is_last_pixel = (x == xend);

            if (run > 0 && (run == 0x2020 || color != prev || is_last_pixel))
            {
                if (run < 33)
                {
                    run -= 1;
                    bytes[p++] = QOI_RUN_8 | run;
                }
                else
                {
                    run -= 33;
                    bytes[p++] = QOI_RUN_16 | run >> 8;
                    bytes[p++] = run;
                }
                run = 0;
            }

            if (color != prev)
            {
                int index_pos = QOI_COLOR_HASH(color) % 64;

                if (index[index_pos] == color)
                {
                    bytes[p++] = QOI_INDEX | index_pos;
                }
                else
                {
                    index[index_pos] = color;

                    int r = color.r - prev.r;
                    int g = color.g - prev.g;
                    int b = color.b - prev.b;
                    int a = color.a - prev.a;

                    if (is_diff(r, g, b, a))
                    {
                        if (is_diff8(r, g, b, a))
                        {
                            bytes[p++] = QOI_DIFF_8 | ((r + 1) << 4) | (g + 1) << 2 | (b + 1);
                        }
                        else if (is_diff16(r, g, b, a))
                        {
                            bytes[p++] = QOI_DIFF_16 | (r + 15);
                            bytes[p++] = ((g + 7) << 4) | (b + 7);
                        }
                        else
                        {
                            bytes[p++] = QOI_DIFF_24 | ((r + 15) >> 1);
                            bytes[p++] = ((r + 15) << 7) | ((g + 15) << 2) | ((b + 15) >> 3);
                            bytes[p++] = ((b + 15) << 5) | (a + 15);
                        }
                    }
                    else
                    {
                        int p0 = p;
                        bytes[p++] = QOI_COLOR;

                        int mask = 0;
                        if (r)
                        {
                            mask |= 8;
                            bytes[p++] = color.r;
                        }
                        if (g)
                        {
                            mask |= 4;
                            bytes[p++] = color.g;
                        }
                        if (b)
                        {
                            mask |= 2;
                            bytes[p++] = color.b;
                        }
                        if (a)
                        {
                            mask |= 1;
                            bytes[p++] = color.a;
                        }

                        bytes[p0] |= mask;
                    }
                }
            }

            prev = color;
        }

        image += stride;
    }

    for (int i = 0; i < QOI_PADDING; i++)
    {
        bytes[p++] = 0;
    }

    *out_len = p;
    return bytes;
}

*/

void qoi_decode(u8* image, const u8* data, size_t size, int width, int height, size_t stride)
{
    Color color(0, 0, 0, 255);
    Color index[64] = { 0 };

    //const u8* end = data + size;
    int run = 0;

    for (int y = 0; y < height; ++y)
    {
        Color* dest = reinterpret_cast<Color*>(image);
        Color* xend = dest + width;

        for ( ; dest < xend; )
        {
            if (run > 0)
            {
                --run;
                *dest++ = color;
            }
            else
            {
                u32 b1 = *data++;

                if ((b1 & QOI_MASK_2) == QOI_INDEX)
                {
                    color = index[b1 ^ QOI_INDEX];
                }
                else if ((b1 & QOI_MASK_3) == QOI_RUN_8)
                {
                    run = (b1 & 0x1f);
                }
                else if ((b1 & QOI_MASK_3) == QOI_RUN_16)
                {
                    u32 b2 = *data++;
                    run = (((b1 & 0x1f) << 8) | (b2)) + 32;
                }
                else if ((b1 & QOI_MASK_2) == QOI_DIFF_8)
                {
                    color.r += ((b1 >> 4) & 0x03) - 1;
                    color.g += ((b1 >> 2) & 0x03) - 1;
                    color.b += ((b1 >> 0) & 0x03) - 1;
                }
                else if ((b1 & QOI_MASK_3) == QOI_DIFF_16)
                {
                    u32 b2 = *data++;
                    color.r += (b1 & 0x1f) - 15;
                    color.g += (b2 >> 4) - 7;
                    color.b += (b2 & 0x0f) - 7;
                }
                else if ((b1 & QOI_MASK_4) == QOI_DIFF_24)
                {
                    u32 b = (b1 << 16) | (data[0] << 8) | data[1];
                    data += 2;
                    color.r += ((b >> 15) & 0x1f) - 15;
                    color.g += ((b >> 10) & 0x1f) - 15;
                    color.b += ((b >>  5) & 0x1f) - 15;
                    color.a += ((b >>  0) & 0x1f) - 15;
                }
                else if ((b1 & QOI_MASK_4) == QOI_COLOR)
                {
                    if (b1 & 8) { color.r = *data++; }
                    if (b1 & 4) { color.g = *data++; }
                    if (b1 & 2) { color.b = *data++; }
                    if (b1 & 1) { color.a = *data++; }
                }

                if (b1 & QOI_UPDATE)
                {
                    index[QOI_COLOR_HASH(color) % 64] = color;
                }

                *dest++ = color;
            }
        }

        image += stride;
    }
}

/*

void qoi_decode(u8* image, const u8* data, size_t size, int width, int height, size_t stride)
{
    Color color(0, 0, 0, 255);
    Color index[64] = { 0 };

    //const u8* end = data + size;
    int run = 0;

    for (int y = 0; y < height; ++y)
    {
        Color* dest = reinterpret_cast<Color*>(image);
        Color* xend = dest + width;

        for ( ; dest < xend; )
        {
            if (run > 0)
            {
                --run;
                *dest++ = color;
            }
            else
            {
                int b1 = *data++;

                if ((b1 & QOI_MASK_2) == QOI_INDEX)
                {
                    color = index[b1 ^ QOI_INDEX];
                }
                else if ((b1 & QOI_MASK_3) == QOI_RUN_8)
                {
                    run = (b1 & 0x1f);
                }
                else if ((b1 & QOI_MASK_3) == QOI_RUN_16)
                {
                    int b2 = *data++;
                    run = (((b1 & 0x1f) << 8) | (b2)) + 32;
                }
                else if ((b1 & QOI_MASK_2) == QOI_DIFF_8)
                {
                    color.r += ((b1 >> 4) & 0x03) - 1;
                    color.g += ((b1 >> 2) & 0x03) - 1;
                    color.b += ( b1       & 0x03) - 1;
                }
                else if ((b1 & QOI_MASK_3) == QOI_DIFF_16)
                {
                    int b2 = *data++;
                    color.r += (b1 & 0x1f) - 15;
                    color.g += (b2 >> 4) - 7;
                    color.b += (b2 & 0x0f) - 7;
                }
                else if ((b1 & QOI_MASK_4) == QOI_DIFF_24)
                {
                    int b2 = *data++;
                    int b3 = *data++;
                    color.r += (((b1 & 0x0f) << 1) | (b2 >> 7)) - 15;
                    color.g +=  ((b2 & 0x7c) >> 2) - 15;
                    color.b += (((b2 & 0x03) << 3) | ((b3 & 0xe0) >> 5)) - 15;
                    color.a +=   (b3 & 0x1f) - 15;
                }
                else if ((b1 & QOI_MASK_4) == QOI_COLOR)
                {
                    if (b1 & 8) { color.r = *data++; }
                    if (b1 & 4) { color.g = *data++; }
                    if (b1 & 2) { color.b = *data++; }
                    if (b1 & 1) { color.a = *data++; }
                }

                if (b1 & QOI_UPDATE)
                {
                    index[QOI_COLOR_HASH(color) % 64] = color;
                }

                *dest++ = color;
            }
        }

        image += stride;
    }
}

*/

#endif // QOI_IMPLEMENTATION
