/*!
 * \file support/module_vtable.h
 * \brief Compatibility shim providing the TVM_MODULE_VTABLE_* macros that
 *        previously lived in <tvm/runtime/module.h>. After the TVM runtime
 *        refactor the macros were moved into TVM's private
 *        src/runtime/vm/module_utils.h, so we vendor the surface mlc-llm uses
 *        here to keep the existing call sites unchanged.
 */
#ifndef MLC_LLM_SUPPORT_MODULE_VTABLE_H_
#define MLC_LLM_SUPPORT_MODULE_VTABLE_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <utility>

namespace mlc {
namespace llm {
namespace module_vtable_detail {

template <typename T>
struct EntryHelper {};

template <typename T, typename R, typename... Args>
struct EntryHelper<R (T::*)(Args...) const> {
  using MemFnType = R (T::*)(Args...) const;
  TVM_FFI_INLINE static void Call(::tvm::ffi::Any* rv, T* self, MemFnType f,
                                  ::tvm::ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ::tvm::ffi::details::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr,
                                        wrapped, args.data(), args.size(), rv);
  }
};

template <typename T, typename R, typename... Args>
struct EntryHelper<R (T::*)(Args...)> {
  using MemFnType = R (T::*)(Args...);
  TVM_FFI_INLINE static void Call(::tvm::ffi::Any* rv, T* self, MemFnType f,
                                  ::tvm::ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ::tvm::ffi::details::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr,
                                        wrapped, args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct EntryHelper<void (T::*)(Args...) const> {
  using MemFnType = void (T::*)(Args...) const;
  TVM_FFI_INLINE static void Call(::tvm::ffi::Any* rv, T* self, MemFnType f,
                                  ::tvm::ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ::tvm::ffi::details::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr,
                                           wrapped, args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct EntryHelper<void (T::*)(Args...)> {
  using MemFnType = void (T::*)(Args...);
  TVM_FFI_INLINE static void Call(::tvm::ffi::Any* rv, T* self, MemFnType f,
                                  ::tvm::ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ::tvm::ffi::details::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr,
                                           wrapped, args.data(), args.size(), rv);
  }
};

}  // namespace module_vtable_detail
}  // namespace llm
}  // namespace mlc

#define TVM_MODULE_VTABLE_BEGIN(TypeKey)                                                  \
  const char* kind() const final { return TypeKey; }                                      \
  ::tvm::ffi::Optional<::tvm::ffi::Function> GetFunction(const ::tvm::ffi::String& _name) \
      override {                                                                          \
    using SelfPtr = std::remove_cv_t<decltype(this)>;                                     \
    ::tvm::ffi::ObjectPtr<::tvm::ffi::Object> _self =                                     \
        ::tvm::ffi::GetObjectPtr<::tvm::ffi::Object>(this);
#define TVM_MODULE_VTABLE_END() \
  return std::nullopt;          \
  }
#define TVM_MODULE_VTABLE_END_WITH_DEFAULT(MemFunc) \
  {                                                 \
    auto f = (MemFunc);                             \
    return (this->*f)(_name);                       \
  }                                                 \
  }
#define TVM_MODULE_VTABLE_ENTRY(Name, MemFunc)                                             \
  if (_name == Name) {                                                                     \
    return ::tvm::ffi::Function::FromPacked(                                               \
        [_self](::tvm::ffi::PackedArgs args, ::tvm::ffi::Any* rv) -> void {                \
          using Helper = ::mlc::llm::module_vtable_detail::EntryHelper<decltype(MemFunc)>; \
          SelfPtr self = static_cast<SelfPtr>(_self.get());                                \
          Helper::Call(rv, self, MemFunc, args);                                           \
        });                                                                                \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                       \
  if (_name == Name) {                                                      \
    return ::tvm::ffi::Function(                                            \
        [_self](::tvm::ffi::PackedArgs args, ::tvm::ffi::Any* rv) -> void { \
          (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);        \
        });                                                                 \
  }

#endif  // MLC_LLM_SUPPORT_MODULE_VTABLE_H_
