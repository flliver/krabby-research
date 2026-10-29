#include "unity.h"
#include "Wire.h"

void setUp() { Wire.reset(); }
void tearDown() { TEST_ASSERT_TRUE(Wire.state().events.empty()); }

static wire_native::Transfer response(uint8_t address, std::initializer_list<uint8_t> bytes = {})
{
    wire_native::Transfer value;
    value.address = address;
    value.bytes = bytes;
    value.reported = static_cast<uint8_t>(bytes.size());
    return value;
}

static void test_addresses_and_interleaved_buses_keep_their_own_data()
{
    TwoWire first, second;
    first.state().transfers = {response(0x6B, {1, 2}), response(0x3D)};
    second.state().transfers = {response(0x6B, {9})};
    first.beginTransmission(0x6B);
    TEST_ASSERT_EQUAL_UINT(1, first.write(0x20));
    TEST_ASSERT_EQUAL_INT(0, first.endTransmission(false));
    TEST_ASSERT_EQUAL_INT(1, second.requestFrom(0x6B, 1));
    TEST_ASSERT_EQUAL_INT(2, first.requestFrom(0x6B, 2));
    TEST_ASSERT_EQUAL_INT(1, first.read());
    TEST_ASSERT_EQUAL_INT(9, second.read());
    TEST_ASSERT_EQUAL_INT(2, first.read());
    first.beginTransmission(0x3D);
    TEST_ASSERT_EQUAL_INT(0, first.endTransmission());
    TEST_ASSERT_TRUE(first.state().transfers.empty());
    TEST_ASSERT_TRUE(second.state().transfers.empty());
    TEST_ASSERT_TRUE(first.state().errors.empty());
    TEST_ASSERT_TRUE(second.state().errors.empty());
    TEST_ASSERT_EQUAL_INT(0, first.available());
    TEST_ASSERT_EQUAL_INT(0, second.available());
}

static void test_repeated_start_and_requested_count_are_recorded()
{
    TwoWire bus;
    bus.state().transfers.push_back(response(0x6B, {42}));
    bus.beginTransmission(0x6B);
    bus.write(0x20);
    bus.endTransmission(false);
    TEST_ASSERT_EQUAL_INT(1, bus.requestFrom(0x6B, 14, true));
    const auto &events = bus.state().events;
    TEST_ASSERT_EQUAL_UINT(4, events.size());
    TEST_ASSERT_EQUAL_STRING("wire.transmit", events[0].name.c_str());
    TEST_ASSERT_EQUAL_INT(0x6B, events[0].args[0]);
    TEST_ASSERT_EQUAL_STRING("wire.write", events[1].name.c_str());
    TEST_ASSERT_EQUAL_INT(0x20, events[1].args[0]);
    TEST_ASSERT_EQUAL_STRING("wire.endTransmission", events[2].name.c_str());
    TEST_ASSERT_EQUAL_INT(0, events[2].args[0]);
    TEST_ASSERT_EQUAL_STRING("wire.request", events[3].name.c_str());
    TEST_ASSERT_TRUE(events[3].args == std::vector<long>({0x6B, 14, 1}));
    TEST_ASSERT_EQUAL_INT(42, bus.read());
    TEST_ASSERT_TRUE(bus.state().errors.empty());
}

static void test_nack_write_failure_and_timeout_are_independent()
{
    TwoWire bus, other;
    auto nack = response(0x40);
    nack.status = 2;
    nack.written = 0;
    auto timeout = response(0x41);
    timeout.timeout = true;
    bus.state().transfers = {nack, timeout, response(0x40)};
    bus.beginTransmission(0x40);
    TEST_ASSERT_EQUAL_UINT(0, bus.write(7));
    TEST_ASSERT_EQUAL_INT(2, bus.endTransmission());
    TEST_ASSERT_FALSE(bus.getWireTimeoutFlag());
    bus.beginTransmission(0x41);
    TEST_ASSERT_EQUAL_INT(0, bus.endTransmission());
    TEST_ASSERT_TRUE(bus.getWireTimeoutFlag());
    TEST_ASSERT_FALSE(other.getWireTimeoutFlag());
    bus.beginTransmission(0x40);
    TEST_ASSERT_EQUAL_INT(0, bus.endTransmission());
    TEST_ASSERT_TRUE(bus.getWireTimeoutFlag());
    bus.clearWireTimeoutFlag();
    TEST_ASSERT_FALSE(bus.getWireTimeoutFlag());
    TEST_ASSERT_TRUE(bus.state().errors.empty());
}

static void test_reported_count_does_not_conceal_buffer_exhaustion()
{
    TwoWire bus;
    auto shortRead = response(0x6B, {4});
    shortRead.reported = 14;
    shortRead.timeout = true;
    bus.state().transfers.push_back(shortRead);
    TEST_ASSERT_EQUAL_INT(14, bus.requestFrom(0x6B, 14));
    TEST_ASSERT_TRUE(bus.getWireTimeoutFlag());
    TEST_ASSERT_EQUAL_INT(1, bus.available());
    TEST_ASSERT_EQUAL_INT(4, bus.read());
    TEST_ASSERT_EQUAL_INT(0, bus.available());
    TEST_ASSERT_EQUAL_INT(-1, bus.read());
    TEST_ASSERT_EQUAL_UINT(1, bus.state().errors.size());
}

static void test_unexpected_operations_fail_and_unconsumed_scripts_remain_visible()
{
    TwoWire bus;
    TEST_ASSERT_EQUAL_UINT(0, bus.write(7));
    TEST_ASSERT_EQUAL_INT(2, bus.endTransmission());
    bus.beginTransmission(0x41);
    TEST_ASSERT_EQUAL_INT(2, bus.endTransmission());
    bus.state().transfers = {response(0x40), response(0x3D)};
    bus.beginTransmission(0x41);
    bus.endTransmission(false);
    bus.requestFrom(0x42, 1);
    TEST_ASSERT_EQUAL_UINT(5, bus.state().errors.size());
    TEST_ASSERT_EQUAL_UINT(1, bus.state().transfers.size());
    TEST_ASSERT_EQUAL_INT(0x3D, bus.state().transfers.front().address);
}

static void test_bus_lifecycle_configuration_and_observer_are_isolated()
{
    wire_native::State state;
    TwoWire bus(state), other;
    std::vector<std::string> observed;
    state.observe = [&observed](const wire_native::Event &event) { observed.push_back(event.name); };
    bus.begin();
    bus.setClock(400000);
    bus.setWireTimeout(10000, true);
    TEST_ASSERT_TRUE(state.begun);
    TEST_ASSERT_EQUAL_UINT32(400000, state.clock);
    TEST_ASSERT_EQUAL_UINT32(10000, state.timeoutMicroseconds);
    TEST_ASSERT_TRUE(state.resetOnTimeout);
    TEST_ASSERT_FALSE(other.state().begun);
    TEST_ASSERT_EQUAL_UINT32(100000, other.state().clock);
    TEST_ASSERT_EQUAL_UINT32(0, other.state().timeoutMicroseconds);
    TEST_ASSERT_FALSE(other.state().resetOnTimeout);
    state.transfers.push_back(response(0x40, {1, 2}));
    bus.requestFrom(0x40, 2);
    bus.end();
    TEST_ASSERT_FALSE(state.begun);
    TEST_ASSERT_EQUAL_INT(0, bus.available());
    TEST_ASSERT_EQUAL_UINT(5, observed.size());
    TEST_ASSERT_EQUAL_STRING("wire.end", observed.back().c_str());
    bus.beginTransmission(0x40); // An unscripted transaction must not survive reset.
    bus.reset();
    TEST_ASSERT_FALSE(state.begun);
    TEST_ASSERT_FALSE(state.timeout);
    TEST_ASSERT_EQUAL_UINT32(100000, state.clock);
    TEST_ASSERT_TRUE(state.events.empty());
    TEST_ASSERT_TRUE(state.errors.empty());
    TEST_ASSERT_TRUE(state.transfers.empty());
    TEST_ASSERT_FALSE(bool(state.observe));
    TEST_ASSERT_EQUAL_INT(0, bus.available());
    TEST_ASSERT_EQUAL_UINT(0, bus.write(1));
    TEST_ASSERT_EQUAL_UINT(1, state.errors.size());
}

static void test_buffered_write_records_each_byte_and_sums_scripted_counts()
{
    TwoWire bus;
    wire_native::Transfer accepted;
    accepted.address = 0x40;
    wire_native::Transfer refused;
    refused.address = 0x41;
    refused.written = 0;
    bus.state().transfers.push_back(accepted);
    bus.state().transfers.push_back(refused);
    const uint8_t bytes[] = {0x01, 0x02, 0x03};

    bus.beginTransmission(0x40);
    TEST_ASSERT_EQUAL_UINT(3, bus.write(bytes, sizeof bytes));
    TEST_ASSERT_EQUAL_INT(0, bus.endTransmission());
    bus.beginTransmission(0x41);
    TEST_ASSERT_EQUAL_UINT(0, bus.write(bytes, sizeof bytes));
    TEST_ASSERT_EQUAL_INT(0, bus.endTransmission());

    const auto &events = bus.state().events;
    TEST_ASSERT_EQUAL_UINT(10, events.size());
    for (size_t index = 0; index < 3; ++index)
    {
        TEST_ASSERT_EQUAL_STRING("wire.write", events[1 + index].name.c_str());
        TEST_ASSERT_EQUAL_INT(bytes[index], events[1 + index].args[0]);
    }
    TEST_ASSERT_TRUE(bus.state().errors.empty());
    TEST_ASSERT_TRUE(bus.state().transfers.empty());
}

namespace {
class RecordingDevice : public wire_native::Device
{
public:
    uint8_t status = 0;
    std::vector<uint8_t> response;
    std::vector<std::vector<uint8_t>> transmitted;
    std::vector<uint8_t> requested;
    uint8_t transmit(const std::vector<uint8_t> &bytes) override
    {
        transmitted.push_back(bytes);
        return status;
    }
    std::vector<uint8_t> receive(uint8_t count) override
    {
        requested.push_back(count);
        return response;
    }
};
}

static void test_attached_device_answers_its_address_and_scripts_serve_the_rest()
{
    TwoWire bus;
    RecordingDevice device;
    device.response = {0xAB, 0xCD};
    bus.state().devices[0x40] = &device;
    wire_native::Transfer scripted;
    scripted.address = 0x41;
    scripted.reported = 1;
    scripted.bytes = {7};
    bus.state().transfers.push_back(scripted);

    bus.beginTransmission(0x40);
    const uint8_t pointer[] = {0x05};
    TEST_ASSERT_EQUAL_UINT(1, bus.write(pointer, sizeof pointer));
    TEST_ASSERT_EQUAL_INT(0, bus.endTransmission(false));
    TEST_ASSERT_EQUAL_INT(2, bus.requestFrom(0x40, 3, true));
    TEST_ASSERT_EQUAL_INT(0xAB, bus.read());
    TEST_ASSERT_EQUAL_INT(0xCD, bus.read());
    TEST_ASSERT_EQUAL_INT(0, bus.available());
    TEST_ASSERT_EQUAL_UINT(1, device.transmitted.size());
    TEST_ASSERT_TRUE(device.transmitted[0] == std::vector<uint8_t>({0x05}));
    TEST_ASSERT_TRUE(device.requested == std::vector<uint8_t>({3}));

    device.status = 5;
    bus.beginTransmission(0x40);
    TEST_ASSERT_EQUAL_INT(5, bus.endTransmission());
    TEST_ASSERT_TRUE(bus.getWireTimeoutFlag());
    TEST_ASSERT_EQUAL_UINT(2, device.transmitted.size());
    TEST_ASSERT_TRUE(device.transmitted[1].empty());

    TEST_ASSERT_EQUAL_INT(1, bus.requestFrom(0x41, 1));
    TEST_ASSERT_EQUAL_INT(7, bus.read());
    TEST_ASSERT_TRUE(bus.state().transfers.empty());
    TEST_ASSERT_TRUE(bus.state().errors.empty());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_addresses_and_interleaved_buses_keep_their_own_data);
    RUN_TEST(test_repeated_start_and_requested_count_are_recorded);
    RUN_TEST(test_nack_write_failure_and_timeout_are_independent);
    RUN_TEST(test_reported_count_does_not_conceal_buffer_exhaustion);
    RUN_TEST(test_unexpected_operations_fail_and_unconsumed_scripts_remain_visible);
    RUN_TEST(test_bus_lifecycle_configuration_and_observer_are_isolated);
    RUN_TEST(test_buffered_write_records_each_byte_and_sums_scripted_counts);
    RUN_TEST(test_attached_device_answers_its_address_and_scripts_serve_the_rest);
    return UNITY_END();
}
