/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/auto_spec_decode.cc
 */

#include <tvm/runtime/nvtx.h>

#include <numeric>

#include "../config.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that first makes a decision on whether to run speculative
 * decoding or normal mode batch decode, and then runs the selected actions.
 */
class AutoSpecDecodeActionObj : public EngineActionObj {
 public:
  explicit AutoSpecDecodeActionObj(Array<EngineAction> spec_decode_actions,
                                   Array<EngineAction> batch_decode_actions,
                                   EngineConfig engine_config)
      : spec_decode_actions_(std::move(spec_decode_actions)),
        batch_decode_actions_(std::move(batch_decode_actions)),
        engine_config_(std::move(engine_config)) {}

  Array<Request> Step(EngineState estate) final {
    int num_running_rsentries = estate->GetRunningRequestStateEntries().size();
    if (num_running_rsentries == 0) {
      return {};
    }

    // Calculate the draft length to use for the next round decode.
    estate->spec_draft_length = CalculateDraftLength(estate, num_running_rsentries);
    ICHECK_GE(estate->spec_draft_length, 0);
    Array<Request> processed_requests;
    // Use speculative decoding when the computed draft length is positive.
    // Otherwise use normal mode batch decode.
    Array<EngineAction> actions =
        estate->spec_draft_length > 0 ? spec_decode_actions_ : batch_decode_actions_;
    for (EngineAction action : actions) {
      processed_requests = action->Step(estate);
    }

    // Reset the draft length.
    estate->spec_draft_length = 0;
    return processed_requests;
  }

 private:
  int CalculateDraftLength(EngineState estate, int num_running_rsentries) {
    // Right now we use the fixed table to select the draft length (only based on
    // the batch size). We will follow up to adopt powerful draft length selection.
    if (num_running_rsentries < 10) {
      return 4;
    } else if (num_running_rsentries < 20) {
      return 3;
    } else if (num_running_rsentries < 30) {
      return 2;
    } else {
      return 0;
    }
  }

  /*! \brief The speculative decode actions. */
  Array<EngineAction> spec_decode_actions_;
  /*! \brief The normal mode decode actions. */
  Array<EngineAction> batch_decode_actions_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
};

EngineAction EngineAction::AutoSpecDecode(std::vector<EngineAction> spec_decode_actions_,
                                          std::vector<EngineAction> batch_decode_actions_,
                                          EngineConfig engine_config) {
  return EngineAction(make_object<AutoSpecDecodeActionObj>(
      Array<EngineAction>(spec_decode_actions_), Array<EngineAction>(batch_decode_actions_),
      std::move(engine_config)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
