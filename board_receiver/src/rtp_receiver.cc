#include "rtp_receiver.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstring>

namespace board_receiver {

UdpRtpReceiver::UdpRtpReceiver() : socket_(-1) {}
UdpRtpReceiver::~UdpRtpReceiver() { close(); }

bool UdpRtpReceiver::open(uint16_t port, int receive_buffer_bytes, int timeout_ms,
                          std::string* error) {
    close();
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) { if (error) *error = std::strerror(errno); return false; }
    if (setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, &receive_buffer_bytes,
                   sizeof(receive_buffer_bytes)) != 0) {
        if (error) *error = std::strerror(errno);
        close();
        return false;
    }
    timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        if (error) *error = std::strerror(errno);
        close();
        return false;
    }
    return true;
}

bool UdpRtpReceiver::receive(ReceivedDatagram* datagram, bool* timed_out,
                             std::string* error) {
    if (timed_out) *timed_out = false;
    if (!datagram || socket_ < 0) { if (error) *error = "receiver is not open"; return false; }
    uint8_t buffer[65536];
    sockaddr_in source;
    socklen_t source_size = sizeof(source);
    const ssize_t size = recvfrom(socket_, buffer, sizeof(buffer), 0,
                                  reinterpret_cast<sockaddr*>(&source), &source_size);
    if (size < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            if (timed_out) *timed_out = true;
            return true;
        }
        if (error) *error = std::strerror(errno);
        return false;
    }
    datagram->bytes.assign(buffer, buffer + size);
    char text[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &source.sin_addr, text, sizeof(text));
    datagram->source = std::string(text) + ":" + std::to_string(ntohs(source.sin_port));
    datagram->received_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return true;
}

void UdpRtpReceiver::close() {
    if (socket_ >= 0) ::close(socket_);
    socket_ = -1;
}

}  // namespace board_receiver
