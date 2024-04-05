/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/encoding.cc
 */
#include "encoding.h"

#include <tvm/runtime/logging.h>

#include <array>

namespace mlc {
namespace llm {

std::string CodepointToUtf8(TCodepoint codepoint) {
  ICHECK(codepoint <= 0x10FFFF) << "Invalid codepoint: " << codepoint;
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

std::string CodepointToPrintable(
    TCodepoint codepoint, const std::unordered_map<TCodepoint, std::string>& custom_escape_map) {
  static const std::unordered_map<TCodepoint, std::string> kCodepointToEscape = {
      {'\'', "\\\'"}, {'\"', "\\\""}, {'\?', "\\\?"}, {'\\', "\\\\"}, {'\a', "\\a"},
      {'\b', "\\b"},  {'\f', "\\f"},  {'\n', "\\n"},  {'\r', "\\r"},  {'\t', "\\t"},
      {'\v', "\\v"},  {'\0', "\\0"},  {'\x1B', "\\e"}};

  if (auto it = custom_escape_map.find(codepoint); it != custom_escape_map.end()) {
    return it->second;
  }

  if (auto it = kCodepointToEscape.find(codepoint); it != kCodepointToEscape.end()) {
    return it->second;
  }

  if (codepoint >= 0x20 && codepoint <= 0x7E) {
    return std::string({static_cast<char>(codepoint)});
  }

  // convert codepoint to hex
  int width = codepoint <= 0xFFFF ? 4 : 8;
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(width) << std::hex << codepoint;
  auto hex = ss.str();
  return codepoint <= 0xFFFF ? "\\u" + hex : "\\U" + hex;
}

std::pair<TCodepoint, int> Utf8ToCodepoint(const char* utf8) {
  const std::array<int8_t, 5> kFirstByteMask = {0x00, 0x7F, 0x1F, 0x0F, 0x07};
  // clang-format off
  const std::array<int, 256> kUtf8Bytes = {
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

  auto bytes = kUtf8Bytes[static_cast<unsigned char>(utf8[0])];
  if (bytes == -1) {
    // invalid utf8
    return {static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8), 0};
  }

  TCodepoint res = static_cast<unsigned char>(utf8[0]) & kFirstByteMask[bytes];
  for (int i = 1; i < bytes; ++i) {
    if (utf8[i] == 0 || (static_cast<unsigned char>(utf8[i]) & 0xC0) != 0x80) {
      // invalid utf8
      return {static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8), 0};
    }
    res = (res << 6) | (static_cast<unsigned char>(utf8[i]) & 0x3F);
  }
  return {res, bytes};
}

std::vector<TCodepoint> Utf8StringToCodepoints(const char* utf8) {
  std::vector<TCodepoint> codepoints;
  while (*utf8 != 0) {
    auto [codepoint, bytes] = Utf8ToCodepoint(utf8);
    if (codepoint == static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8)) {
      return {codepoint};
    }
    codepoints.push_back(codepoint);
    utf8 += bytes;
  }
  return codepoints;
}

int HexCharToInt(char c) {
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

std::pair<TCodepoint, int> Utf8OrEscapeToCodepoint(
    const char* utf8, const std::unordered_map<std::string, TCodepoint>& custom_escape_map) {
  static const std::unordered_map<std::string, TCodepoint> kEscapeToCodepoint = {
      {"\\\'", '\''}, {"\\\"", '\"'}, {"\\\?", '\?'}, {"\\\\", '\\'}, {"\\a", '\a'},
      {"\\b", '\b'},  {"\\f", '\f'},  {"\\n", '\n'},  {"\\r", '\r'},  {"\\t", '\t'},
      {"\\v", '\v'},  {"\\0", '\0'},  {"\\e", '\x1B'}};
  if (utf8[0] != '\\') {
    return Utf8ToCodepoint(utf8);
  }

  auto escape_sequence = std::string(utf8, 2);
  if (auto it = custom_escape_map.find(escape_sequence); it != custom_escape_map.end()) {
    return {it->second, 2};
  }
  if (auto it = kEscapeToCodepoint.find(escape_sequence); it != kEscapeToCodepoint.end()) {
    return {it->second, 2};
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
      return {static_cast<TCodepoint>(CharHandlingError::kInvalidEscape), 0};
    }
    return {codepoint, len + 2};
  } else if (utf8[1] == 'u' || utf8[1] == 'U') {
    // 4- or 8-digit hex
    int len = utf8[1] == 'u' ? 4 : 8;
    int32_t codepoint = 0;

    for (int i = 0; i < len; ++i) {
      auto digit = HexCharToInt(utf8[i + 2]);
      if (digit == -1) {
        return {static_cast<TCodepoint>(CharHandlingError::kInvalidEscape), 0};
      }
      codepoint = codepoint * 16 + digit;
    }
    return {codepoint, len + 2};
  } else {
    return {static_cast<TCodepoint>(CharHandlingError::kInvalidEscape), 0};
  }
}

}  // namespace llm
}  // namespace mlc
