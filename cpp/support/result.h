/*!
 * Copyright (c) 2023-2025 by Contributors
 * \file support/result.h
 * \brief The header for the Result class in MLC LLM.
 */
#ifndef MLC_LLM_SUPPORT_RESULT_H_
#define MLC_LLM_SUPPORT_RESULT_H_

#include <tvm/runtime/logging.h>

#include <optional>
#include <string>

namespace mlc {
namespace llm {

/*!
 * \brief The result class in MLC LLM.
 * Each instance is either an okay value or an error.
 * \tparam T The okay value type of the result.
 * \tparam E The error type of the result.
 */
template <typename T, typename E = std::string>
class Result {
 public:
  /*! \brief Create a result with an okay value. */
  static Result Ok(T value) {
    Result result;
    result.ok_value_ = std::move(value);
    return result;
  }
  /*! \brief Create a result with an error value. */
  static Result Error(E error) {
    Result result;
    result.err_value_ = std::move(error);
    return result;
  }
  /*! \brief Check if the result is okay or not. */
  bool IsOk() const { return ok_value_.has_value(); }
  /*! \brief Check if the result is an error or not. */
  bool IsErr() const { return err_value_.has_value(); }
  /*!
   * \brief Unwrap the result and return the okay value.
   * Throwing exception if it is an error.
   * \note This function returns the ok value by moving, so a Result can be unwrapped only once.
   */
  T Unwrap() {
    TVM_FFI_ICHECK(ok_value_.has_value()) << "Cannot unwrap result on an error value.";
    TVM_FFI_ICHECK(!unwrapped_) << "Cannot unwrap a Result instance twice.";
    unwrapped_ = true;
    return std::move(ok_value_.value());
  }
  /*!
   * \brief Unwrap the result and return the error value.
   * Throwing exception if it is an okay value.
   * \note This function returns the error value by moving, so a Result can be unwrapped only once.
   */
  E UnwrapErr() {
    TVM_FFI_ICHECK(err_value_.has_value()) << "Cannot unwrap result on an okay value.";
    TVM_FFI_ICHECK(!unwrapped_) << "Cannot unwrap a Result instance twice.";
    unwrapped_ = true;
    return std::move(err_value_.value());
  }

 private:
  /*! \brief A boolean flag indicating if the result is okay or error. */
  bool unwrapped_ = false;
  /*! \brief The internal optional okay value. */
  std::optional<T> ok_value_;
  /*! \brief The internal optional error value. */
  std::optional<E> err_value_;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_RESULT_H_
