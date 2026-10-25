#pragma once

#include "sim_serial.hpp"
#include "src/controller/board_role.h"
#include "src/controller/board_role_election.hpp"

namespace krabby {
namespace sim {

void resetHostFrontImuPresence();
void setHostBoardHasFrontImu(BoardId board, bool present);
bool hostBoardHasFrontImu(BoardId board);

void setActiveEepromBoard(BoardId board);
BoardId activeEepromBoard();

struct HostBoardRoleElectionContext
{
    BoardId boardId = BoardId::kFront;
    BoardRoleElectionState state;
    BoardRole electedRole = ROLE_UNKNOWN;
    bool finished = false;
};

void hostBoardRoleElectionBegin(HostBoardRoleElectionContext &ctx);
bool hostBoardRoleElectionStep(HostBoardRoleElectionContext &ctx);

void runThreeBoardRoleElectionCooperative(
    BoardRole rolesOut[3],
    unsigned long maxSteps = 500);

}  // namespace sim
}  // namespace krabby
