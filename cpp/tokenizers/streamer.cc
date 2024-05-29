/*!
 *  Copyright (c) 2023 by Contributors
 * \file streamer.cc
 */

#include "streamer.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <string>

#include "tokenizers.h"

namespace mlc {
namespace llm {

/****************** TextStreamer ******************/

TVM_REGISTER_OBJECT_TYPE(TextStreamerObj);

TextStreamerObj::TextStreamerObj(Tokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}

TextStreamer::TextStreamer(Tokenizer tokenizer) {
  data_ = make_object<TextStreamerObj>(std::move(tokenizer));
}

std::string TextStreamerObj::Put(const std::vector<int32_t>& delta_tokens) {
  CHECK(!finished_) << "`put` is not expected to be invoked after finish.";
  if (delta_tokens.empty()) {
    return "";
  }

  std::string ret;
  // We process delta tokens one by one.
  for (int32_t delta_token : delta_tokens) {
    // push to pending tokens.
    pending_tokens_.push_back(delta_token);

    // all_tokens = prefix_tokens_ + pending_tokens_
    std::vector<int32_t> all_tokens;
    all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
    all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(), prefix_tokens_.end());
    all_tokens.insert(all_tokens.end(), pending_tokens_.begin(), pending_tokens_.end());

    // Decode prefix_tokens_ and all_tokens.
    std::string prefix_str = prefix_tokens_.empty() ? "" : tokenizer_->Decode(prefix_tokens_);
    std::string full_str = tokenizer_->Decode(all_tokens);

    std::string validated_str;
    std::vector<int32_t> new_pending_tokens;
    if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
      // Case 1. prefix_str is a prefix of `full_str`.
      // validated_str = full_str[len(prefix_str):]
      validated_str = full_str.substr(prefix_str.length());
      // Pop UTF-8 replacement character from the back of pending tokens.
      // - The UTF-8 replacement character take 3 chars.
      // - A valid UTF-8 has 4 chars at most.
      //   So there will be at most 3 tokens popped.
      while (!pending_tokens_.empty() &&                         //
             static_cast<int>(new_pending_tokens.size()) < 3 &&  //
             validated_str.length() >= 3 &&                      //
             validated_str.compare(validated_str.length() - 3, /*n=*/3, kReplacementCharacter) ==
                 0) {
        new_pending_tokens.push_back(pending_tokens_.back());
        pending_tokens_.pop_back();
        validated_str = validated_str.substr(0, validated_str.length() - 3);
      }
    } else {
      // Case 2. prefix_str is not a prefix of `full_str`.
      // Pop pending tokens from the back.
      // - Pop until prefix_str is indeed a prefix of full_str.
      // - A valid UTF-8 has 4 chars at most.
      //   So there will be at most 3 tokens popped.
      // - If there are no more than 3 pending tokens, skip popping.
      //   This is because it is impossible to make full_str contain
      //   prefix_str without popping all the pending tokens.
      if (static_cast<int>(pending_tokens_.size()) < 3) {
        continue;
      }
      bool get_valid_full_str = false;
      while (!pending_tokens_.empty() && static_cast<int>(new_pending_tokens.size()) < 3) {
        new_pending_tokens.push_back(pending_tokens_.back());
        pending_tokens_.pop_back();
        all_tokens.pop_back();
        full_str = tokenizer_->Decode(all_tokens);
        if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
          get_valid_full_str = true;
          break;
        }
      }

      if (get_valid_full_str) {
        // We find a full_str which starts from prefix_str.
        // So we return the sliced full string without the prefix.
        validated_str = full_str.substr(prefix_str.length());
      } else {
        // We cannot find a full_str which starts from prefix_str by
        // popping 3 tokens.
        // In this case, the remaining pending tokens are invalid UTF-8
        // characters already, so we return the decoded pending tokens.
        validated_str = tokenizer_->Decode(pending_tokens_);
      }
    }

    if (!pending_tokens_.empty()) {
      // Set the new prefix.
      prefix_tokens_ = pending_tokens_;
    }
    std::reverse(new_pending_tokens.begin(), new_pending_tokens.end());
    pending_tokens_ = new_pending_tokens;
    ret += validated_str;
  }
  return ret;
}

std::string TextStreamerObj::Finish() {
  // all_tokens = prefix_tokens_ + pending_tokens_
  std::vector<int32_t> all_tokens;
  all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
  all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(), prefix_tokens_.end());
  all_tokens.insert(all_tokens.end(), pending_tokens_.begin(), pending_tokens_.end());

  // Decode prefix_tokens_ and all_tokens.
  std::string prefix_str = prefix_tokens_.empty() ? "" : tokenizer_->Decode(prefix_tokens_);
  std::string full_str = all_tokens.empty() ? "" : tokenizer_->Decode(all_tokens);

  finished_ = true;
  if (full_str.compare(0, prefix_str.length(), prefix_str) == 0) {
    // Case 1. prefix_str is a prefix of `full_str`.
    return full_str.substr(prefix_str.length());
  } else {
    // Case 2. prefix_str is not a prefix of `full_str`.
    // In this case, the remaining pending tokens are invalid UTF-8
    // characters already, so we return the decoded pending tokens.
    return tokenizer_->Decode(pending_tokens_);
  }
}

TVM_REGISTER_GLOBAL("mlc.tokenizers.TextStreamer").set_body_typed([](Tokenizer tokenizer) {
  return TextStreamer(std::move(tokenizer));
});

TVM_REGISTER_GLOBAL("mlc.tokenizers.TextStreamerPut")
    .set_body_typed([](TextStreamer text_streamer, const IntTuple& delta_tokens) {
      return text_streamer->Put({delta_tokens->data, delta_tokens->data + delta_tokens->size});
    });

TVM_REGISTER_GLOBAL("mlc.tokenizers.TextStreamerFinish")
    .set_body_method<TextStreamer>(&TextStreamerObj::Finish);

/****************** StopStrHandler ******************/

TVM_REGISTER_OBJECT_TYPE(StopStrHandlerObj);

/*! \brief Create the KMP partial match table for the input string. */
inline std::vector<int> CreatePartialMatchTable(const String& str) {
  int length = str.length();
  std::vector<int> partial_match_table = {-1};
  partial_match_table.reserve(length);
  for (int i = 1; i < length; ++i) {
    int ptr = partial_match_table[i - 1];
    while (ptr != -1 && str.at(ptr) != str.at(i - 1)) {
      ptr = partial_match_table[ptr];
    }
    partial_match_table.push_back(ptr + 1);
  }
  return partial_match_table;
}

StopStrHandlerObj::StopStrHandlerObj(Array<String> stop_strs,
                                     const std::vector<std::string>& token_table)
    : stop_strs_(std::move(stop_strs)), token_table_(token_table) {
  int num_stop_strs = stop_strs_.size();
  cur_match_lengths_.resize(num_stop_strs, 0);

  // Create the KMP partial match table for each stop string.
  partial_match_tables_.reserve(num_stop_strs);
  for (const String& stop_str : stop_strs_) {
    CHECK(!stop_str.empty()) << "Stop string cannot be empty.";
    partial_match_tables_.push_back(CreatePartialMatchTable(stop_str));
  }
}

std::vector<int32_t> StopStrHandlerObj::Put(int32_t token_id) {
  // Return the input token id if there is no stop string.
  if (stop_strs_.empty()) {
    return {token_id};
  }

  CHECK(!stop_triggered_) << "Cannot put new token when already stopped.";

  ICHECK_LT(token_id, static_cast<int>(token_table_.size()));
  const std::string& token = token_table_[token_id];
  pending_token_ids_.push_back(token_id);
  pending_token_lengths_.push_back(token.length());

  std::vector<int32_t> return_token_ids;

  for (char ch : token) {
    // The earliest starting point of stop string.
    int stop_starting_pos = std::numeric_limits<int>::max();
    // The cutoff length that can be safely return.
    int cutoff_length = std::numeric_limits<int>::max();
    // The maximum matched length.
    int max_match_length = 0;

    for (int str_id = 0; str_id < static_cast<int>(stop_strs_.size()); ++str_id) {
      // - Run one step of KMP algorithm.
      const std::vector<int>& partial_match_table = partial_match_tables_[str_id];
      int& cur_match_length = cur_match_lengths_[str_id];
      while (cur_match_length != -1 && ch != stop_strs_[str_id].at(cur_match_length)) {
        cur_match_length = partial_match_table[cur_match_length];
      }
      ++cur_match_length;

      // Case 1. The stop string is matched.
      if (cur_match_length == stop_strs_[str_id].length()) {
        stop_triggered_ = true;
        stop_starting_pos =
            std::min(stop_starting_pos,
                     pending_string_len_ + 1 - static_cast<int>(stop_strs_[str_id].length()));
        continue;
      }

      // Case 2. The stop string is not matched.
      // - Get the cutoff length that can be safely return.
      ICHECK_GE(pending_string_len_ + 1, cur_match_length);
      cutoff_length = std::min(cutoff_length, pending_string_len_ + 1 - cur_match_length);
      // - Get the updated pending string length.
      max_match_length = std::max(max_match_length, cur_match_length);
    }

    // Collect the token ids that can be safely cut off and returned.
    if (stop_triggered_) {
      cutoff_length = stop_starting_pos;
    }
    ICHECK_NE(cutoff_length, std::numeric_limits<int>::max());
    ICHECK_GE(cutoff_length, 0);
    int cum_length = 0;
    while (!pending_token_ids_.empty() &&
           cum_length + pending_token_lengths_.front() <= cutoff_length) {
      cum_length += pending_token_lengths_.front();
      return_token_ids.push_back(pending_token_ids_.front());
      pending_token_ids_.erase(pending_token_ids_.begin());
      pending_token_lengths_.erase(pending_token_lengths_.begin());
    }
    if (stop_triggered_) {
      return return_token_ids;
    }

    ICHECK_LE(cum_length, cutoff_length);
    // `cum_length` is the prefix length what we actually cut off.
    pending_string_len_ = (cutoff_length - cum_length) + max_match_length;
  }
  return return_token_ids;
}

StopStrHandler::StopStrHandler(Array<String> stop_strs,
                               const std::vector<std::string>& token_table) {
  data_ = make_object<StopStrHandlerObj>(std::move(stop_strs), token_table);
}

TVM_REGISTER_GLOBAL("mlc.tokenizers.StopStrHandler")
    .set_body_typed([](Array<String> stop_strs, const Tokenizer& tokenizer) {
      return StopStrHandler(std::move(stop_strs), tokenizer->PostProcessedTokenTable());
    });

TVM_REGISTER_GLOBAL("mlc.tokenizers.StopStrHandlerPut")
    .set_body_typed([](StopStrHandler handler, int token_id) {
      std::vector<int32_t> delta_tokens = handler->Put(token_id);
      return IntTuple(delta_tokens.begin(), delta_tokens.end());
    });

TVM_REGISTER_GLOBAL("mlc.tokenizers.StopStringHandlerFinish")
    .set_body_typed([](StopStrHandler handler) {
      std::vector<int32_t> remaining_token_ids = handler->Finish();
      return IntTuple(remaining_token_ids.begin(), remaining_token_ids.end());
    });

TVM_REGISTER_GLOBAL("mlc.tokenizers.StopStrHandlerStopTriggered")
    .set_body_method<StopStrHandler>(&StopStrHandlerObj::StopTriggered);

}  // namespace llm
}  // namespace mlc
