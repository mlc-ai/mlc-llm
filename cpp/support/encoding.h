/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/encoding.h
 * \brief Encoding and decoding from/to UTF-8 and escape sequence to/from codepoints.
 */
#ifndef MLC_LLM_SERVE_ENCODING_H_
#define MLC_LLM_SERVE_ENCODING_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace mlc {
namespace llm {

/*! \brief Represents a unicode codepoint. */
using TCodepoint = int32_t;

/*!
 * \brief Convert a codepoint to a UTF-8 string.
 * \param codepoint The codepoint.
 * \return The UTF-8 string.
 */
std::string CodepointToUtf8(TCodepoint codepoint);

/*!
 * \brief Convert a codepoint to a printable string. If the codepoint is not printable, it will be
 * escaped. By default the function support escape sequences in C ("\n", "\t", "\u0123"). User can
 * specify more escape sequences using custom_escape_map.
 * \param codepoint The codepoint.
 * \param custom_escape_map A map from codepoint to escape sequence. If the codepoint is in the map,
 * it will be escaped using the corresponding escape sequence. e.g. {'-', "\\-"}.
 * \return The printable string.
 */
std::string CodepointToPrintable(
    TCodepoint codepoint,
    const std::unordered_map<TCodepoint, std::string>& custom_escape_map = {});

/*!
 * \brief Represents an error when handling characters. Will be returned as a special TCodepoint
 * value.
 */
enum class CharHandlingError : TCodepoint {
  /*! \brief The UTF-8 string is invalid. */
  kInvalidUtf8 = -10,
  /*! \brief The escape sequence is invalid. */
  kInvalidEscape = -11,
};

/*!
 * \brief Convert a UTF-8 string to a codepoint.
 * \param utf8 The UTF-8 string.
 * \return The codepoint and the number of bytes consumed. If the UTF-8 string is invalid, the
 * function returns (CharHandlingError::kInvalidUtf8, 0).
 */
std::pair<TCodepoint, int> Utf8ToCodepoint(const char* utf8);

std::vector<TCodepoint> Utf8StringToCodepoints(const char* utf8);

/*!
 * \brief Convert a UTF-8 string or an escape sequence to a codepoint. By default the function
 * supports escape sequences in C ("\n", "\t", "\u0123"). User can specify more escape sequences
 * using custom_escape_map.
 * \param utf8 The UTF-8 string or the escape sequence.
 * \param custom_escape_map A map from escape sequence to codepoint. If the escape sequence is in
 * the map, it will be converted to the corresponding codepoint. e.g. {"\\-", '-'}.
 * \return The codepoint and the number of bytes consumed. If the UTF-8 string or the escape
 * sequence is invalid, the function returns
 * (CharHandlingError::kInvalidUtf8 or CharHandlingError::kInvalidEscape, 0).
 */
std::pair<TCodepoint, int> Utf8OrEscapeToCodepoint(
    const char* utf8, const std::unordered_map<std::string, TCodepoint>& custom_escape_map = {});

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENCODING_H_
