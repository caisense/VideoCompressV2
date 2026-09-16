#ifndef BOARD_RECEIVER_BOARD_DISPLAY_H_
#define BOARD_RECEIVER_BOARD_DISPLAY_H_

#include <string>

#include "mpp_hevc_decoder.h"

namespace board_receiver {

class BoardDisplay {
public:
    BoardDisplay(bool fullscreen, bool rotate_ccw);
    ~BoardDisplay();
    static void prepareWaylandEnvironment();
    bool open(std::string* error);
    bool show(const DecodedFrame& frame, std::string* error);
    void close();
private:
    bool fullscreen_;
    bool rotate_ccw_;
    bool opened_;
};

}  // namespace board_receiver
#endif

