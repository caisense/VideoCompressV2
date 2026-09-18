#ifndef ROI_H265_TRANSPORT_LOCAL_TX_RATE_PUBLISHER_H_
#define ROI_H265_TRANSPORT_LOCAL_TX_RATE_PUBLISHER_H_

#include <stdint.h>

#include <string>

namespace roi_h265 {

// Best-effort local telemetry only. Failure to publish must never interrupt
// the network video path.
class LocalTxRatePublisher {
public:
    explicit LocalTxRatePublisher(const std::string &socket_path);
    ~LocalTxRatePublisher();

    void publish(uint32_t wire_bps);

private:
    std::string socket_path_;
    int socket_fd_;
};

}  // namespace roi_h265

#endif  // ROI_H265_TRANSPORT_LOCAL_TX_RATE_PUBLISHER_H_
