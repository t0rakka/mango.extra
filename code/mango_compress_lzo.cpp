
#include "lzo/minilzo.h"

// ----------------------------------------------------------------------------
// lzo
// ----------------------------------------------------------------------------

namespace lzo
{

    size_t bound(size_t size)
    {
        return size + (size / 16) + 128;
    }

    CompressionStatus compress(Memory dest, ConstMemory source, int level)
    {
        MANGO_UNREFERENCED(level);

        Buffer work(LZO1X_MEM_COMPRESS);

        lzo_uint dst_len = (lzo_uint)dest.size;
        int x = lzo1x_1_compress(source.address, lzo_uint(source.size),
            dest.address, &dst_len, work);

        CompressionStatus status;

        if (x != LZO_E_OK)
        {
            status.setError("[lzo] compression failed.");
        }

        status.size = size_t(dst_len);
        return status;
	}

    CompressionStatus decompress(Memory dest, ConstMemory source)
    {
        lzo_uint dst_len = (lzo_uint)dest.size;
        int x = lzo1x_decompress(source.address, lzo_uint(source.size),
            dest.address, &dst_len, nullptr);

        CompressionStatus status;

        if (x != LZO_E_OK)
        {
            status.setError("[lzo] decompression failed.");
        }

        status.size = dest.size;
        return status;
    }

} // namespace lzo
