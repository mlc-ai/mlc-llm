/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/event_trace_recorder.cc
 */
#include "event_trace_recorder.h"

#include <picojson.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

using tvm::ffi::String;

namespace detail {

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

}  // namespace detail

TVM_REGISTER_OBJECT_TYPE(EventTraceRecorderObj);

/*! \brief The implementation of event trace recorder. */
class EventTraceRecorderImpl : public EventTraceRecorderObj {
 public:
  void AddEvent(const String& request_id, const std::string& event) final {
    double event_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      AddEventInternal(request_id, event, event_time);
    }
  }

  void AddEvent(const Array<String>& request_ids, const std::string& event) final {
    double event_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();  // in seconds

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const String& request_id : request_ids) {
        AddEventInternal(request_id, event, event_time);
      }
    }
  }

  std::string DumpJSON() final {
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> local_events;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      local_events = events_;
    }

    auto fcmp_events = [](const std::pair<int64_t, picojson::value>& lhs,
                          const std::pair<int64_t, picojson::value>& rhs) {
      return lhs.first < rhs.first;
    };

    picojson::array event_array;
    for (const std::string& request_id : request_id_in_order_) {
      std::vector<std::pair<std::string, double>> event_pairs = local_events.at(request_id);
      std::vector<std::pair<int64_t, picojson::value>> events_to_sort;
      events_to_sort.reserve(event_pairs.size());
      for (int i = 0; i < static_cast<int>(event_pairs.size()); ++i) {
        std::string event = event_pairs[i].first;
        double event_time = event_pairs[i].second;
        std::string name;
        std::string phase;
        if (event.compare(0, 6, "start ") == 0) {
          // Duration begin.
          name = event.substr(6);
          phase = "B";
        } else if (event.compare(0, 7, "finish ") == 0) {
          // Duration end.
          name = event.substr(7);
          phase = "E";
        } else {
          // Instant event.
          name = event;
          phase = "i";
        }
        int64_t event_time_in_us = static_cast<int64_t>(event_time * 1e6);

        picojson::object event_json;
        event_json["name"] = picojson::value(name);
        event_json["ph"] = picojson::value(phase);
        event_json["ts"] = picojson::value(event_time_in_us);
        event_json["pid"] = picojson::value(static_cast<int64_t>(1));
        event_json["tid"] = picojson::value(request_id);

        events_to_sort.push_back({event_time_in_us, picojson::value(event_json)});
      }
      std::sort(events_to_sort.begin(), events_to_sort.end(), fcmp_events);
      for (auto [timestamp, event] : events_to_sort) {
        event_array.push_back(std::move(event));
      }
    }
    return picojson::value(event_array).serialize();
  }

  TVM_DECLARE_BASE_OBJECT_INFO(EventTraceRecorderImpl, EventTraceRecorderObj);

 private:
  /*! \brief The internal impl of AddEvent, taking the event time as input. */
  void AddEventInternal(const std::string& request_id, const std::string& event,
                        double event_time) {
    if (std::find(request_id_in_order_.begin(), request_id_in_order_.end(), request_id) ==
        request_id_in_order_.end()) {
      request_id_in_order_.push_back(request_id);
    }
    int event_cnt = event_counter_[{request_id, event}]++;
    events_[request_id].push_back({event + " (" + std::to_string(event_cnt) + ")", event_time});
  }

  /*! \brief The mutex ensuring only one thread can access critical regions. */
  std::mutex mutex_;

  /************** Critical Regions **************/
  /*! \brief The request ids in time order. Each id only appears once. */
  std::vector<std::string> request_id_in_order_;
  /*! \brief The number of a certain event for a request. */
  std::unordered_map<std::pair<std::string, std::string>, int, detail::PairHash> event_counter_;
  /*! \brief The event list of each request together with the timestamps. */
  std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> events_;
};

EventTraceRecorder EventTraceRecorder::Create() {
  return EventTraceRecorder(tvm::ffi::make_object<EventTraceRecorderImpl>());
}

TVM_FFI_REGISTER_GLOBAL("mlc.serve.EventTraceRecorder").set_body_typed([]() {
  return EventTraceRecorder::Create();
});

TVM_FFI_REGISTER_GLOBAL("mlc.serve.EventTraceRecorderAddEvent")
    .set_body_typed([](const EventTraceRecorder& trace_recorder, const String& request_id,
                       const std::string& event) { trace_recorder->AddEvent(request_id, event); });

TVM_FFI_REGISTER_GLOBAL("mlc.serve.EventTraceRecorderDumpJSON")
    .set_body_method(&EventTraceRecorderObj::DumpJSON);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
