#ifndef BOARD_RECEIVER_LOCAL_TX_RATE_RECEIVER_H_
#define BOARD_RECEIVER_LOCAL_TX_RATE_RECEIVER_H_

#include <stdint.h>

#include <string>

namespace board_receiver {

class LocalTxRateReceiver {
public:
    LocalTxRateReceiver();
    ~LocalTxRateReceiver();

    bool open(const std::string& socket_path, std::string* error);
    bool receiveLatest(uint32_t* wire_bps);
    void close();

private:
    int socket_fd_;
    std::string socket_path_;
};

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_LOCAL_TX_RATE_RECEIVER_H_
