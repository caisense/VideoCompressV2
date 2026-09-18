#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "local_tx_rate_receiver.h"
#include "transport/local_tx_rate_publisher.h"
#include "test_support.h"

int main() {
    const std::string path = "/tmp/board_tx_rate_test_" +
                             std::to_string(static_cast<long long>(getpid())) + ".sock";
    board_receiver::LocalTxRateReceiver receiver;
    std::string error;
    CHECK(receiver.open(path, &error));

    roi_h265::LocalTxRatePublisher publisher(path);
    publisher.publish(123400U);
    uint32_t wire_bps = 0;
    for (int attempt = 0; attempt < 20 && wire_bps == 0; ++attempt) {
        receiver.receiveLatest(&wire_bps);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(wire_bps == 123400U);

    publisher.publish(80000U);
    publisher.publish(150000U);
    CHECK(receiver.receiveLatest(&wire_bps));
    CHECK(wire_bps == 150000U);
    receiver.close();
    CHECK(access(path.c_str(), F_OK) != 0);
    return 0;
}
