#include "local_tx_rate_receiver.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>

namespace board_receiver {

LocalTxRateReceiver::LocalTxRateReceiver() : socket_fd_(-1) {}

LocalTxRateReceiver::~LocalTxRateReceiver() { close(); }

bool LocalTxRateReceiver::open(const std::string& socket_path, std::string* error) {
    close();
    sockaddr_un address;
    if (socket_path.empty() || socket_path.size() >= sizeof(address.sun_path)) {
        if (error) *error = "invalid local TX rate socket path";
        return false;
    }
    socket_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        if (error) *error = std::string("local TX rate socket failed: ") + std::strerror(errno);
        return false;
    }
    const int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        if (error) *error = std::string("local TX rate nonblocking mode failed: ") +
                            std::strerror(errno);
        close();
        return false;
    }

    std::memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    unlink(socket_path.c_str());
    if (bind(socket_fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
        if (error) *error = std::string("local TX rate bind failed: ") + std::strerror(errno);
        close();
        return false;
    }
    socket_path_ = socket_path;
    if (error) error->clear();
    return true;
}

bool LocalTxRateReceiver::receiveLatest(uint32_t* wire_bps) {
    if (socket_fd_ < 0 || !wire_bps) return false;
    bool received = false;
    for (;;) {
        uint8_t message[32];
        const ssize_t size = recv(socket_fd_, message, sizeof(message), MSG_DONTWAIT);
        if (size < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (size == 8 && message[0] == 'T' && message[1] == 'X' &&
            message[2] == 'R' && message[3] == '1') {
            *wire_bps = (static_cast<uint32_t>(message[4]) << 24) |
                        (static_cast<uint32_t>(message[5]) << 16) |
                        (static_cast<uint32_t>(message[6]) << 8) |
                        static_cast<uint32_t>(message[7]);
            received = true;
        }
    }
    return received;
}

void LocalTxRateReceiver::close() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
    if (!socket_path_.empty()) {
        unlink(socket_path_.c_str());
        socket_path_.clear();
    }
}

}  // namespace board_receiver
