#include "transport/local_tx_rate_publisher.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>

namespace roi_h265 {

LocalTxRatePublisher::LocalTxRatePublisher(const std::string &socket_path)
    : socket_path_(socket_path), socket_fd_(-1) {}

LocalTxRatePublisher::~LocalTxRatePublisher() {
    if (socket_fd_ >= 0) close(socket_fd_);
}

void LocalTxRatePublisher::publish(uint32_t wire_bps) {
    if (socket_path_.empty()) return;
    if (socket_fd_ < 0) socket_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) return;

    sockaddr_un address;
    std::memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) return;
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);

    uint8_t message[8] = {'T', 'X', 'R', '1',
        static_cast<uint8_t>(wire_bps >> 24),
        static_cast<uint8_t>(wire_bps >> 16),
        static_cast<uint8_t>(wire_bps >> 8),
        static_cast<uint8_t>(wire_bps)};
    (void)sendto(socket_fd_, message, sizeof(message), MSG_DONTWAIT,
                 reinterpret_cast<const sockaddr*>(&address), sizeof(address));
}

}  // namespace roi_h265
