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

TVM_REGISTER_GLOBAL("mlc.TextStreamer").set_body_typed([](Tokenizer tokenizer) {
  return TextStreamer(std::move(tokenizer));
});

TVM_REGISTER_GLOBAL("mlc.TextStreamerPut")
    .set_body_typed([](TextStreamer text_streamer, const IntTuple& delta_tokens) {
      return text_streamer->Put({delta_tokens->data, delta_tokens->data + delta_tokens->size});
    });

TVM_REGISTER_GLOBAL("mlc.TextStreamerFinish")
    .set_body_method<TextStreamer>(&TextStreamerObj::Finish);

/****************** StopStrHandler ******************/

TVM_REGISTER_OBJECT_TYPE(StopStringHandlerObj);

StopStringHandlerObj::StopStringHandlerObj(std::vector<std::string> stop_strs)
    : stop_strs_(std::move(stop_strs)) {
  max_stop_str_length_ = 0;
  for (const std::string& stop_str : stop_strs_) {
    max_stop_str_length_ = std::max(max_stop_str_length_, static_cast<int>(stop_str.length()));
  }
  if (!stop_strs_.empty()) {
    CHECK_GT(max_stop_str_length_, 0);
  }
}

/*!
 * \brief Find the valid UTF-8 cutoff point in the input string
 * in front of the input position, so that the substring in front
 * of the cutoff point is a valid UTF-8 string.
 */
inline int FindUTF8CutoffPosition(const std::string& str, int pos) {
  static constexpr const char* error_msg = "The input string is invalid UTF-8 encoded.";
  ICHECK_GE(pos, 0);
  ICHECK_LT(pos, static_cast<int>(str.length()));

  for (int i = 0; i < 4 && pos - i > 0; ++i) {
    unsigned char byte = static_cast<unsigned char>(str[pos - i - 1]);
    if (byte <= 0x7F) {
      // Single-byte character (ASCII)
      ICHECK_EQ(i, 0);
      return pos;
    } else if (byte >= 0xC0 && byte <= 0xDF) {
      ICHECK_LE(i, 1);
      return i == 1 ? pos : pos - i - 1;
    } else if (byte >= 0xE0 && byte <= 0xEF) {
      ICHECK_LE(i, 2);
      return i == 2 ? pos : pos - i - 1;
    } else if (byte >= 0xF0 && byte <= 0xF7) {
      return i == 3 ? pos : pos - i - 1;
    }
  }
  ICHECK(false) << "Cannot reach here.";
  throw;
}

std::string StopStringHandlerObj::Put(std::string input_delta_str) {
  CHECK(!stop_triggered_) << "`put` is not expected to be invoked after stopped.";
  if (stop_triggered_) {
    return "";
  }

  // - Decode the new tokens and get the delta string.
  //   Concatenate to the suffix.
  if (input_delta_str.empty()) {
    return "";
  }
  if (stop_strs_.empty()) {
    return input_delta_str;
  }
  pending_str_ += input_delta_str;

  // - Check if any stop strings appear.
  size_t earliest_occurrence_pos = std::string::npos;
  for (const std::string& stop_str : stop_strs_) {
    earliest_occurrence_pos = std::min(earliest_occurrence_pos, pending_str_.find(stop_str));
  }

  // - Return the prefix if any stop strings appear.
  if (earliest_occurrence_pos != std::string::npos) {
    stop_triggered_ = true;
    return pending_str_.substr(0, earliest_occurrence_pos);
  }
  // - Update the suffix.
  int pending_str_length = static_cast<int>(pending_str_.length());
  if (pending_str_length > max_stop_str_length_ - 1) {
    // Note that we cannot return UTF-8 invalid strings.
    // So we always need to find a cutoff position so that
    // the cut substring is UTF-8 valid.
    int cutoff_pos =
        FindUTF8CutoffPosition(pending_str_, pending_str_length - (max_stop_str_length_ - 1));
    // - Return suffix[:cutoff_pos]
    // - Update suffix to suffix[cutoff_pos:]
    std::string output_delta_str = pending_str_.substr(0, cutoff_pos);
    ICHECK_GE(cutoff_pos, 0);
    pending_str_ = pending_str_.substr(cutoff_pos);
    return output_delta_str;
  } else {
    // Suffix is not long enough for return.
    return "";
  }
}

std::string StopStringHandlerObj::Finish() {
  std::string ret = pending_str_;
  pending_str_ = "";
  return ret;
}

StopStringHandler::StopStringHandler(std::vector<std::string> stop_strs) {
  data_ = make_object<StopStringHandlerObj>(std::move(stop_strs));
}

TVM_REGISTER_GLOBAL("mlc.StopStringHandler").set_body_typed([](Array<String> stop_strs) {
  return StopStringHandler({stop_strs.begin(), stop_strs.end()});
});

TVM_REGISTER_GLOBAL("mlc.StopStringHandlerPut")
    .set_body_typed([](StopStringHandler handler, String input_delta_str) {
      return handler->Put(std::move(input_delta_str));
    });

TVM_REGISTER_GLOBAL("mlc.StopStringHandlerFinish")
    .set_body_method<StopStringHandler>(&StopStringHandlerObj::Finish);

TVM_REGISTER_GLOBAL("mlc.StopStringHandlerStopTriggered")
    .set_body_method<StopStringHandler>(&StopStringHandlerObj::StopTriggered);

}  // namespace llm
}  // namespace mlc
