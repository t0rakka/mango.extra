/*
    MANGO Multimedia Development Platform
    Copyright (C) 2012-2025 Twilight Finland 3D Oy Ltd. All rights reserved.
*/
#include <mango/mango.hpp>

using namespace mango;
using namespace mango::filesystem;

void test(const std::string& pathname, const std::string& filename, const std::string& password, u32 expected)
{
    Path path(pathname + "/", password);
    File file(path, filename);
    u32 checksum = crc32(0, file);

    if (checksum == expected)
    {
        printLine("{:<24} : PASSED", pathname);
    }
    else
    {
        printLine("{:<24} : FAILED {:#x}", pathname, checksum);
    }
}

int main()
{
    test("../data/deflate.zip", "mipsIV32.pdf", "", 0x69dc3b95);
    test("../data/bzip2.zip", "mipsIV32.pdf", "", 0x69dc3b95);
    test("../data/lzma.zip", "mipsIV32.pdf", "", 0x69dc3b95);
    test("../data/ppmd.zip", "mipsIV32.pdf", "", 0x69dc3b95);
    test("../data/bzip2_crypto.zip", "station.jpg", "rapa1234", 0xafce3b8d);
    // TODO: not supported
    //test("../data/bzip2_aes256.zip", "station.jpg", "rapa1234", 0xafce3b8d);
    //test("../data/deflate64.zip", "mipsIV32.pdf", "", 0x69dc3b95);
}
