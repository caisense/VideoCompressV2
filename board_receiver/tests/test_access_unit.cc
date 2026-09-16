#include "test_support.h"

#include "access_unit_assembler.h"

int main() {
    using namespace board_receiver;
    HevcRtpDepacketizer depacketizer;
    AccessUnitAssembler assembler;
    const uint32_t timestamp = 9000;
    const uint8_t types[] = {32, 33, 34, 19};
    AssemblyResult assembled;
    for (size_t i = 0; i < 4; ++i) {
        const RtpPacket packet = makePacket(static_cast<uint16_t>(100 + i), timestamp,
                                            i == 3, nal(types[i], static_cast<uint8_t>(i)));
        assembled = assembler.feed(packet, packetProfile(packet),
                                   depacketizer.feed(packet), false);
    }
    CHECK(assembled.ready && !assembled.dropped);
    CHECK(assembled.access_unit.completeRandomAccessPoint());
    CHECK(assembled.access_unit.bytes.size() == 28);

    RtpPacket start = makePacket(200, 10000, false, nal(1, 1));
    assembled = assembler.feed(start, packetProfile(start), depacketizer.feed(start), false);
    CHECK(!assembled.ready);
    RtpPacket end = makePacket(202, 10000, true, nal(1, 2));
    depacketizer.reset();
    assembled = assembler.feed(end, packetProfile(end), depacketizer.feed(end), true);
    CHECK(assembled.ready && assembled.dropped);
    return EXIT_SUCCESS;
}

