#include "test_support.h"

#include "hevc_depacketizer.h"

int main() {
    using namespace board_receiver;
    HevcRtpDepacketizer depacketizer;

    DepacketizedPayload result = depacketizer.feed(
        makePacket(10, 100, true, nal(32, 0xaa)));
    CHECK(result.valid && result.nal_types.size() == 1);
    CHECK(result.nal_types[0] == 32 && result.bytes.size() == 7);

    std::vector<uint8_t> ap;
    ap.push_back(48 << 1); ap.push_back(1);
    const std::vector<uint8_t> first = nal(33, 0xbb);
    const std::vector<uint8_t> second = nal(34, 0xcc);
    appendBe16(&ap, static_cast<uint16_t>(first.size()));
    ap.insert(ap.end(), first.begin(), first.end());
    appendBe16(&ap, static_cast<uint16_t>(second.size()));
    ap.insert(ap.end(), second.begin(), second.end());
    result = depacketizer.feed(makePacket(11, 101, true, ap));
    CHECK(result.valid && result.nal_types.size() == 2);
    CHECK(result.nal_types[0] == 33 && result.nal_types[1] == 34);

    std::vector<uint8_t> bad_ap;
    bad_ap.push_back(48 << 1); bad_ap.push_back(1);
    appendBe16(&bad_ap, 20); bad_ap.push_back(1);
    CHECK(!depacketizer.feed(makePacket(12, 101, true, bad_ap)).valid);

    std::vector<uint8_t> fu_start;
    fu_start.push_back(49 << 1); fu_start.push_back(1); fu_start.push_back(0x80 | 19);
    fu_start.push_back(0xde); fu_start.push_back(0xad);
    result = depacketizer.feed(makePacket(65535, 102, false, fu_start));
    CHECK(result.valid && result.fu_in_progress && result.nal_types[0] == 19);

    std::vector<uint8_t> fu_end;
    fu_end.push_back(49 << 1); fu_end.push_back(1); fu_end.push_back(0x40 | 19);
    fu_end.push_back(0xbe); fu_end.push_back(0xef);
    result = depacketizer.feed(makePacket(0, 102, true, fu_end));
    CHECK(result.valid && !result.fu_in_progress && result.bytes.size() == 2);

    result = depacketizer.feed(makePacket(100, 104, false, fu_start));
    CHECK(result.valid && result.fu_in_progress);
    std::vector<uint8_t> fu_middle;
    fu_middle.push_back(49 << 1); fu_middle.push_back(1); fu_middle.push_back(19);
    fu_middle.push_back(0xca); fu_middle.push_back(0xfe);
    result = depacketizer.feed(makePacket(101, 104, false, fu_middle));
    CHECK(result.valid && result.fu_in_progress && result.bytes.size() == 2);
    result = depacketizer.feed(makePacket(102, 104, true, fu_end));
    CHECK(result.valid && !result.fu_in_progress);

    depacketizer.feed(makePacket(20, 103, false, fu_start));
    result = depacketizer.feed(makePacket(22, 103, true, fu_end));
    CHECK(!result.valid);
    return EXIT_SUCCESS;
}
