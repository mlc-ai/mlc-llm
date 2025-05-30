/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/event_trace_recorder.h
 * \brief The event trace recorder for requests in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_EVENT_TRACE_RECORDER_H_
#define MLC_LLM_SERVE_EVENT_TRACE_RECORDER_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/object.h>

#include <string>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;
using tvm::ffi::Array;
using tvm::ffi::String;

/*! \brief The event trace recorder for requests. */
class EventTraceRecorderObj : public Object {
 public:
  /*!
   * \brief Record a event for the input request in the trace recorder.
   * \param request_id The subject request of the event.
   * \param event The event in a string name.
   * It can have one of the following patterns:
   * - "start xxx", which marks the start of event "xxx",
   * - "finish xxx", which marks the finish of event "xxx",
   * - "yyy", which marks the instant event "yyy".
   * The "starts" and "finishes" will be automatically paired in the trace recorder.
   */
  virtual void AddEvent(const String& request_id, const std::string& event) = 0;

  /*! \brief Record a event for the list of input requests. */
  virtual void AddEvent(const Array<String>& request_ids, const std::string& event) = 0;

  /*! \brief Dump the logged events in Chrome Trace Event Format in JSON string. */
  virtual std::string DumpJSON() = 0;

  static constexpr const char* _type_key = "mlc.serve.EventTraceRecorder";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EventTraceRecorderObj, Object);
};

/*!
 * \brief Managed reference to EventTraceRecorderObj.
 * \sa EventTraceRecorderObj
 */
class EventTraceRecorder : public ObjectRef {
 public:
  /*! \brief Create an event trace recorder. */
  static EventTraceRecorder Create();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(EventTraceRecorder, ObjectRef,
                                                    EventTraceRecorderObj);
};

/****************** Helper macro ******************/

/*! \brief Record a event for the input request or list or requests. */
#define RECORD_EVENT(trace_recorder, request_ids, event)  \
  if (trace_recorder.defined()) {                         \
    trace_recorder.value()->AddEvent(request_ids, event); \
  }

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_EVENT_TRACE_RECORDER_H_
