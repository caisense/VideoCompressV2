#include "rga_core_process.h"

IM_STATUS process_image_rga_core(rga_buffer_t src, rga_buffer_t dst, rga_buffer_t pat,
                                im_rect src_rect, im_rect dst_rect, im_rect pat_rect,
                                int usage, int core_mask) {
    im_opt_t options = {};
    options.core = core_mask;
    return improcess(src, dst, pat, src_rect, dst_rect, pat_rect,
                     -1, nullptr, &options, usage);
}
