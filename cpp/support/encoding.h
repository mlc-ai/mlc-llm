/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/encoding.h
 * \brief Encoding and decoding from/to UTF-8 and escape sequence to/from codepoints.
 */
#ifndef MLC_LLM_SUPPORT_ENCODING_H_
#define MLC_LLM_SUPPORT_ENCODING_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlc {
namespace llm {

/*! \brief Represents a unicode codepoint. */
using TCodepoint = int32_t;

/*!
 * \brief Handle the utf-8 first byte.
 * \returns (is_valid, total_number_of_bytes, initial_codepoint).
 */
std::tuple<bool, int, TCodepoint> HandleUTF8FirstByte(uint8_t byte);

/*!
 * \brief Print a codepoint to a UTF-8 string.
 * \param codepoint The codepoint.
 * \return The UTF-8 string.
 */
std::string PrintAsUTF8(TCodepoint codepoint);

/*!
 * \brief Print a codepoint to a escaped string. If the codepoint is not printable, it will be
 * escaped. By default the function support escape sequences in C ("\n", "\t", "\u0123"). User can
 * specify more escape sequences using additional_escape_map.
 * \param codepoint The codepoint.
 * \param additional_escape_map A map from codepoint to escape sequence. If the codepoint is in the
 * map, it will be escaped using the corresponding escape sequence. e.g. {{'-', "\\-"}}. \return The
 * printable string.
 */
std::string PrintAsEscaped(
    TCodepoint codepoint,
    const std::unordered_map<TCodepoint, std::string>& additional_escape_map = {});

/*!
 * \brief Print the given char to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string PrintAsEscaped(uint8_t raw_char);

/*!
 * \brief Print the given string to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string PrintAsEscaped(std::string raw_str);

/*!
 * \brief Represents an error when handling characters. Will be returned as a special TCodepoint
 * value.
 */
enum CharHandlingError : TCodepoint {
  /*! \brief The UTF-8 string is invalid. */
  kInvalidUTF8 = -10,
  /*! \brief The escape sequence is invalid. */
  kInvalidEscape = -11,
};

/*!
 * \brief The method to handle invalid UTF-8 sequence.
 */
enum class UTF8ErrorPolicy {
  /*! \brief Return an error codepoint when an error is encountered. */
  kReturnInvalid,
  /*! \brief Skip the error and continue parsing. */
  kReturnByte,
};

/*!
 * \brief Parse the first codepoint in a UTF-8 string.
 * \param utf8 The UTF-8 string.
 * \return The codepoint and new pointer. If the UTF-8 string is invalid, and the error policy is
 * kReturnInvalid, the function returns (CharHandlingError::kInvalidUTF8, input char pointer).
 */
std::pair<TCodepoint, const char*> ParseNextUTF8(
    const char* utf8, UTF8ErrorPolicy error_policy = UTF8ErrorPolicy::kReturnInvalid);

/*!
 * \brief Parse all codepoints in a UTF-8 string.
 * \param utf8 The UTF-8 string.
 * \return All codepoints. If the UTF-8 string is invalid, and the error policy is
 * kReturnInvalid, the function returns {CharHandlingError::kInvalidUTF8}.
 */
std::vector<TCodepoint> ParseUTF8(const char* utf8,
                                  UTF8ErrorPolicy error_policy = UTF8ErrorPolicy::kReturnInvalid);

/*!
 * \brief Parse the first codepoint from a UTF-8 string. Also checks escape sequences and converts
 * the escaped char to its original value.
 * \param utf8 The UTF-8 string or the escape sequence.
 * \param additional_escape_map A map from escape sequence to codepoint. If the escape sequence is
 * in the map, it will be converted to the corresponding codepoint. e.g. {{"\\-", '-'}}.
 * \return The codepoint and the new pointer. If the UTF-8 string or the escape sequence is
 * invalid, and the error policy is kReturnInvalid, the function returns
 * (CharHandlingError::kInvalidUTF8, input char pointer).
 */
std::pair<TCodepoint, const char*> ParseNextUTF8OrEscaped(
    const char* utf8,
    const std::unordered_map<std::string, TCodepoint>& additional_escape_map = {});

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_ENCODING_H_
