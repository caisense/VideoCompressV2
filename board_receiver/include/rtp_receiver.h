#ifndef BOARD_RECEIVER_RTP_RECEIVER_H_
#define BOARD_RECEIVER_RTP_RECEIVER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace board_receiver {

struct ReceivedDatagram {
    std::vector<uint8_t> bytes;
    std::string source;
    int64_t received_ms;
};

class UdpRtpReceiver {
public:
    UdpRtpReceiver();
    ~UdpRtpReceiver();
    bool open(uint16_t port, int receive_buffer_bytes, int timeout_ms,
              std::string* error);
    bool receive(ReceivedDatagram* datagram, bool* timed_out, std::string* error);
    void close();

private:
    int socket_;
};

}  // namespace board_receiver
#endif

