/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/encoding.cc
 */
#include "encoding.h"

#include <tvm/runtime/logging.h>

#include <array>

namespace mlc {
namespace llm {

std::string PrintAsUTF8(TCodepoint codepoint) {
  TVM_FFI_ICHECK(codepoint <= 0x10FFFF) << "Invalid codepoint: " << codepoint;
  std::string utf8;
  if (codepoint <= 0x7F) {
    // 1-byte sequence
    utf8 += static_cast<char>(codepoint);
  } else if (codepoint <= 0x7FF) {
    // 2-byte sequence
    utf8 += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else if (codepoint <= 0xFFFF) {
    // 3-byte sequence
    utf8 += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    // 4-byte sequence
    utf8 += static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  }
  return utf8;
}

std::string PrintAsEscaped(
    TCodepoint codepoint,
    const std::unordered_map<TCodepoint, std::string>& additional_escape_map) {
  static const std::unordered_map<TCodepoint, std::string> kCodepointToEscape = {
      {'\'', "\\\'"}, {'\"', "\\\""}, {'\?', "\\\?"}, {'\\', "\\\\"}, {'\a', "\\a"},
      {'\b', "\\b"},  {'\f', "\\f"},  {'\n', "\\n"},  {'\r', "\\r"},  {'\t', "\\t"},
      {'\v', "\\v"},  {'\0', "\\0"},  {'\x1B', "\\e"}};

  if (auto it = additional_escape_map.find(codepoint); it != additional_escape_map.end()) {
    return it->second;
  }

  if (auto it = kCodepointToEscape.find(codepoint); it != kCodepointToEscape.end()) {
    return it->second;
  }

  if (codepoint >= 0x20 && codepoint <= 0x7E) {
    return std::string({static_cast<char>(codepoint)});
  }

  // convert codepoint to hex
  char prefix = codepoint <= 0xFF ? 'x' : codepoint <= 0xFFFF ? 'u' : 'U';
  int width = codepoint <= 0xFF ? 2 : codepoint <= 0xFFFF ? 4 : 8;
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(width) << std::hex << codepoint;
  auto hex = ss.str();
  return std::string("\\") + prefix + hex;
}

std::string PrintAsEscaped(uint8_t raw_char) { return PrintAsEscaped(raw_char); }

std::string PrintAsEscaped(std::string raw_str) {
  std::string res;
  auto codepoints = ParseUTF8(raw_str.c_str(), UTF8ErrorPolicy::kReturnByte);
  for (auto c : codepoints) {
    res += PrintAsEscaped(c);
  }
  return res;
}

std::tuple<bool, int, TCodepoint> HandleUTF8FirstByte(uint8_t byte) {
  static const std::array<int8_t, 5> kFirstByteMask = {0x00, 0x7F, 0x1F, 0x0F, 0x07};
  // clang-format off
  static const std::array<int, 256> kUtf8Bytes = {
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     4,  4,  4,  4,  4,  4,  4,  4, -1, -1, -1, -1, -1, -1, -1, -1,
  };
  // clang-format on
  auto num_bytes = kUtf8Bytes[static_cast<uint8_t>(byte)];
  if (num_bytes == -1) {
    return {false, 0, 0};
  }
  return {true, num_bytes, byte & kFirstByteMask[num_bytes]};
}

std::pair<TCodepoint, const char*> ParseNextUTF8(const char* utf8, UTF8ErrorPolicy error_policy) {
  auto [accepted, num_bytes, res] = HandleUTF8FirstByte(utf8[0]);
  if (accepted) {
    for (int i = 1; i < num_bytes; ++i) {
      if (utf8[i] == 0 || (static_cast<uint8_t>(utf8[i]) & 0xC0) != 0x80) {
        // invalid utf8
        accepted = false;
        break;
      }
      res = (res << 6) | (static_cast<uint8_t>(utf8[i]) & 0x3F);
    }
  }

  if (!accepted) {
    // invalid utf8
    if (error_policy == UTF8ErrorPolicy::kReturnInvalid) {
      return {CharHandlingError::kInvalidUTF8, utf8};
    } else {
      return {static_cast<unsigned char>(utf8[0]), utf8 + 1};
    }
  }

  return {res, utf8 + num_bytes};
}

std::vector<TCodepoint> ParseUTF8(const char* utf8, UTF8ErrorPolicy error_policy) {
  std::vector<TCodepoint> codepoints;
  while (*utf8 != 0) {
    TCodepoint codepoint;
    std::tie(codepoint, utf8) = ParseNextUTF8(utf8, error_policy);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      return {codepoint};
    }
    codepoints.push_back(codepoint);
  }
  return codepoints;
}

inline int HexCharToInt(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    return -1;
  }
}

std::pair<TCodepoint, const char*> ParseNextUTF8OrEscaped(
    const char* utf8, const std::unordered_map<std::string, TCodepoint>& additional_escape_map) {
  static const std::unordered_map<std::string, TCodepoint> kEscapeToCodepoint = {
      {"\\\'", '\''}, {"\\\"", '\"'}, {"\\\?", '\?'}, {"\\\\", '\\'}, {"\\a", '\a'},
      {"\\b", '\b'},  {"\\f", '\f'},  {"\\n", '\n'},  {"\\r", '\r'},  {"\\t", '\t'},
      {"\\v", '\v'},  {"\\0", '\0'},  {"\\e", '\x1B'}};
  if (utf8[0] != '\\') {
    return ParseNextUTF8(utf8, UTF8ErrorPolicy::kReturnInvalid);
  }

  auto escape_sequence = std::string(utf8, 2);
  if (auto it = additional_escape_map.find(escape_sequence); it != additional_escape_map.end()) {
    return {it->second, utf8 + 2};
  }
  if (auto it = kEscapeToCodepoint.find(escape_sequence); it != kEscapeToCodepoint.end()) {
    return {it->second, utf8 + 2};
  }

  if (utf8[1] == 'x') {
    // arbitrary length hex
    int len = 0;
    int32_t codepoint = 0;
    while (true) {
      auto digit = HexCharToInt(utf8[2 + len]);
      if (digit == -1) {
        break;
      }
      codepoint = codepoint * 16 + digit;
      ++len;
    }
    if (len == 0) {
      return {CharHandlingError::kInvalidEscape, utf8};
    }
    return {codepoint, utf8 + len + 2};
  } else if (utf8[1] == 'u' || utf8[1] == 'U') {
    // 4- or 8-digit hex
    int len = utf8[1] == 'u' ? 4 : 8;
    int32_t codepoint = 0;

    for (int i = 0; i < len; ++i) {
      auto digit = HexCharToInt(utf8[i + 2]);
      if (digit == -1) {
        return {CharHandlingError::kInvalidEscape, utf8};
      }
      codepoint = codepoint * 16 + digit;
    }
    return {codepoint, utf8 + len + 2};
  } else {
    return {CharHandlingError::kInvalidEscape, utf8};
  }
}

}  // namespace llm
}  // namespace mlc
