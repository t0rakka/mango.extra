/*
*/
#include <mango/mango.hpp>

#define QOI_IMPLEMENTATION
#include "qoi.h"

using namespace mango;
using namespace mango::image;

void print(const char* name, const char* comment, u64 time0, u64 time1, u64 time2, size_t size)
{
    u64 encode = time1 - time0;
    u64 decode = time2 - time1;

    printf("%s %5d.%d     %5d.%d     %6d  %s\n", name, 
        int(encode / 1000), int(encode % 1000) / 100,
        int(decode / 1000), int(decode % 1000) / 100,
        int(size / 1024), comment);
}

void test_qoi(const char* name, Surface s)
{
    u64 time0 = Time::us();

    size_t length;
    u8* encode_ptr = qoi_encode(s.image, s.stride, s.width, s.height, &length);

    u64 time1 = Time::us();

    int w = s.width;
    int h = s.height;
    Bitmap temp(w, h, s.format);

    qoi_decode(temp.image, encode_ptr, length, w, h, w * 4);

#if 0
    temp.save("result.png");
#endif

    free(encode_ptr);

    u64 time2 = Time::us();
    print(name, "", time0, time1, time2, length);
}

void test_qoi_zstd(const char* name, Surface s)
{
    u64 time0 = Time::us();

    size_t length;
    u8* encode_ptr = qoi_encode(s.image, s.stride, s.width, s.height, &length);

    ConstMemory memory(encode_ptr, (size_t)length);
    size_t bound = zstd::bound(memory.size);
    Buffer compressed(bound);
    size_t size = zstd::compress(compressed, memory, 2);

    u64 time1 = Time::us();

    Buffer decompressed(length);
    zstd::decompress(decompressed, ConstMemory(compressed.data(), size));

    int w = s.width;
    int h = s.height;
    Bitmap temp(w, h, s.format);

    qoi_decode(temp.image, decompressed.data(), decompressed.size(), w, h, w * 4);

#if 0
    temp.save("result.png");
#endif

    free(encode_ptr);

    u64 time2 = Time::us();
    print(name, "", time0, time1, time2, size);
}

void test_qoi_tile(const char* name, Surface s)
{
    u64 time0 = Time::us();

#if 1
    constexpr u32 tile = 64;
#else
    constexpr u32 tile = 64000;
#endif

    const int xs = div_ceil(s.width, tile);
    const int ys = div_ceil(s.height, tile);

    std::vector<Memory> encode_memory(xs * ys);

    ConcurrentQueue q;

    for (int y = 0; y < ys; ++y)
    {
        for (int x = 0; x < xs; ++x)
        {
            Surface rect(s, x * tile, y * tile, tile, tile);
            int idx = y * xs + x;

            q.enqueue([rect, idx, &encode_memory]
            {
                size_t length;
                u8* encode_ptr = qoi_encode(rect.image, rect.stride, rect.width, rect.height, &length);
                encode_memory[idx] = Memory(encode_ptr, length);
            });
        }
    }

    q.wait();

    size_t size = 0;
    for (auto memory : encode_memory)
    {
        size += memory.size;
    }

    u64 time1 = Time::us();

    Bitmap temp(s.width, s.height, s.format);

    for (int y = 0; y < ys; ++y)
    {
        for (int x = 0; x < xs; ++x)
        {
            Surface rect(temp, x * tile, y * tile, tile, tile);
            int idx = y * xs + x;

            q.enqueue([rect, idx, &encode_memory]
            {
                int w = rect.width;
                int h = rect.height;
                Memory memory = encode_memory[idx];

                qoi_decode(rect.image, memory.address, memory.size, w, h, rect.stride);

                free(memory.address);
            });
        }
    }

    q.wait();

#if 0
    temp.save("result.png");
#endif

    u64 time2 = Time::us();
    print(name, "", time0, time1, time2, size);
}

void test_zstd(const char* name, Surface s)
{
    u64 time0 = Time::us();

    ConstMemory memory(s.image, s.width * s.height * 4);
    size_t bound = zstd::bound(memory.size);
    Buffer compressed(bound);
    size_t size = zstd::compress(compressed, memory, 2);

    u64 time1 = Time::us();

    Buffer decompressed(memory.size);
    zstd::decompress(decompressed, ConstMemory(compressed.data(), size));

    u64 time2 = Time::us();
    print(name, "", time0, time1, time2, size);
}

void test_lz4(const char* name, Surface s)
{
    u64 time0 = Time::us();

    ConstMemory memory(s.image, s.width * s.height * 4);
    size_t bound = lz4::bound(memory.size);
    Buffer compressed(bound);
    size_t size = lz4::compress(compressed, memory, 6);

    u64 time1 = Time::us();

    Buffer decompressed(memory.size);
    lz4::decompress(decompressed, ConstMemory(compressed.data(), size));

    u64 time2 = Time::us();
    print(name, "", time0, time1, time2, size);
}

void test_format(const char* name, Surface s, const std::string& extension, bool lossless)
{
    u64 time0 = Time::us();

    MemoryStream output;
    ImageEncoder encoder(extension);
    if (encoder.isEncoder())
    {
        ImageEncodeOptions options;
        //options.multithread = false;
        //options.simd = false;
        encoder.encode(output, s, options);
    }

    u64 time1 = Time::us();

    ImageDecodeOptions options;
    //options.multithread = false;
    //options.simd = false;
    Bitmap bitmap(output, extension, s.format, options);

    u64 time2 = Time::us();

    const char* comment = lossless ? "" : "<-- lossy";
    ::print(name, comment, time0, time1, time2, output.size());
}

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        printf("Too few arguments. usage: <filename.jpg>\n");
        exit(1);
    }

    std::string filename = argv[1];
    Bitmap bitmap(filename, Format(32, Format::UNORM, Format::RGBA, 8, 8, 8, 8));

    printf("\n");
    printf("image: %d x %d (%6d KB )\n", bitmap.width, bitmap.height, int(bitmap.width * bitmap.height * 4 / 1024));
    printf("----------------------------------------------\n");
    printf("         encode(ms)  decode(ms)   size(KB)    \n");
    printf("----------------------------------------------\n");

    test_qoi     ("qoi:      ", bitmap);
    test_qoi_zstd("qoi+zstd: ", bitmap);
    test_qoi_tile("qoi+tile: ", bitmap);
    test_zstd    ("zstd:     ", bitmap);
    test_lz4     ("lz4:      ", bitmap);
    test_format  ("png:      ", bitmap, ".png", true);
    test_format  ("zpng:     ", bitmap, ".zpng", true);
    test_format  ("jpg:      ", bitmap, ".jpg", false);
    test_format  ("webp:     ", bitmap, ".webp", false);
    test_format  ("qoi:      ", bitmap, ".qoi", true);
    test_format  ("toi:      ", bitmap, ".toi", true);
}
