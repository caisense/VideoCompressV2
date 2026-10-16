#ifndef ROI_H265_RGA_CORE_PROCESS_H
#define ROI_H265_RGA_CORE_PROCESS_H

#include <stddef.h>

#include "im2d.h"

#ifdef __cplusplus
extern "C" {
#endif

IM_STATUS process_image_rga_core(rga_buffer_t src, rga_buffer_t dst, rga_buffer_t pat,
                                im_rect src_rect, im_rect dst_rect, im_rect pat_rect,
                                int usage, int core_mask);

#ifdef __cplusplus
}
#endif

#endif
