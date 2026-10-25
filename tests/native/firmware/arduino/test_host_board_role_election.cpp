#include "unity.h"

#include "EEPROM.h"
#include "host_board_role.hpp"
#include "sim_clock.hpp"
#include "sim_serial.hpp"
#include "src/controller/board_role_election.hpp"

using krabby::sim::BoardId;
using krabby::sim::HostBoardRoleElectionContext;
using krabby::sim::SerialPortId;
using krabby::sim::SimSerialPort;
using krabby::sim::hostBoardHasFrontImu;
using krabby::sim::hostBoardRoleElectionBegin;
using krabby::sim::hostBoardRoleElectionStep;
using krabby::sim::runThreeBoardRoleElectionCooperative;
using krabby::sim::serialPort;
using krabby::sim::setActiveEepromBoard;

namespace {

std::string popUsbLine(BoardId board)
{
    SimSerialPort &usb = serialPort(board, SerialPortId::kUsb);
    if (!usb.txLineAvailable()) return {};
    return usb.popTxLine();
}

}  // namespace

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_three_board_election_assigns_front_left_right()
{
    BoardRole roles[3] = {ROLE_UNKNOWN, ROLE_UNKNOWN, ROLE_UNKNOWN};
    runThreeBoardRoleElectionCooperative(roles);

    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, roles[0]);
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, roles[1]);
    TEST_ASSERT_EQUAL_INT(ROLE_RIGHT, roles[2]);
}

static void test_eeprom_role_hint_printed_before_election()
{
    setActiveEepromBoard(BoardId::kLeft);
    EEPROM.update(BOARD_ROLE_EEPROM_ADDR, BOARD_ROLE_EEPROM_MAGIC);
    EEPROM.update(BOARD_ROLE_EEPROM_ADDR + 1, static_cast<uint8_t>(ROLE_LEFT));

    HostBoardRoleElectionContext ctx{};
    ctx.boardId = BoardId::kLeft;
    hostBoardRoleElectionBegin(ctx);
    hostBoardRoleElectionStep(ctx);

    TEST_ASSERT_EQUAL_STRING("--- SYNC ---", popUsbLine(BoardId::kLeft).c_str());
    TEST_ASSERT_EQUAL_STRING("ROLE_HINT: LEFT", popUsbLine(BoardId::kLeft).c_str());
}

static void test_i2c_host_alone_becomes_front_after_election_window()
{
    TEST_ASSERT_TRUE(hostBoardHasFrontImu(BoardId::kFront));

    HostBoardRoleElectionContext ctx{};
    ctx.boardId = BoardId::kFront;
    hostBoardRoleElectionBegin(ctx);

    while (!ctx.finished)
    {
        hostBoardRoleElectionStep(ctx);
        krabby::sim::simClock().advance(BOARD_ROLE_SYNC_INTERVAL_MS);
    }

    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, ctx.electedRole);
    setActiveEepromBoard(BoardId::kFront);
    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, loadBoardRoleHintFromEeprom());
}

static void test_isolated_board_without_imu_times_out_unknown()
{
    krabby::sim::setHostBoardHasFrontImu(BoardId::kFront, false);

    HostBoardRoleElectionContext ctx{};
    ctx.boardId = BoardId::kFront;
    hostBoardRoleElectionBegin(ctx);

    while (!ctx.finished)
    {
        hostBoardRoleElectionStep(ctx);
        krabby::sim::simClock().advance(BOARD_ROLE_SYNC_INTERVAL_MS);
    }

    TEST_ASSERT_EQUAL_INT(ROLE_UNKNOWN, ctx.electedRole);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_three_board_election_assigns_front_left_right);
    RUN_TEST(test_eeprom_role_hint_printed_before_election);
    RUN_TEST(test_i2c_host_alone_becomes_front_after_election_window);
    RUN_TEST(test_isolated_board_without_imu_times_out_unknown);
    return UNITY_END();
}
