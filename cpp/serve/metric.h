/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/metric.h
 * \brief The data structure maintaining the metrics of serving engine/requests.
 */
#ifndef MLC_LLM_SERVE_METRIC_H_
#define MLC_LLM_SERVE_METRIC_H_

#include <picojson.h>

#include <string>

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The class for metric tracking in MLC.
 * - Each metric has a label string which can be empty.
 * - We maintain the number of updates (`count`) and the sum of updated values (`sum`).
 * - We support warmup. When `warmup` is false, the first update will be discarded.
 */
struct Metric {
  std::string label;
  double sum = 0.0;
  int64_t count = 0;
  bool warmed_up = false;

  explicit Metric(bool warmed_up = false, std::string label = "")
      : label(std::move(label)), warmed_up(warmed_up) {}

  /*! \brief Update the metric with given value. */
  void Update(double value) {
    if (warmed_up) {
      sum += value;
      count += 1;
    } else {
      warmed_up = true;
    }
  }

  /*! \brief Set the metric with the given value. */
  void Set(double value) {
    if (warmed_up) {
      sum = value;
      count = 1;
    } else {
      warmed_up = true;
    }
  }

  /*! \brief Reset the metric. */
  void Reset(bool warmed_up = false) {
    this->sum = 0.0;
    this->count = 0;
    this->warmed_up = warmed_up;
  }

  /*! \brief Overloading "+=" for quick update. */
  Metric& operator+=(double value) {
    this->Update(value);
    return *this;
  }

  /*! \brief Dump the metric as JSON. */
  picojson::object AsJSON() const {
    picojson::object config;
    config["label"] = picojson::value(label);
    config["sum"] = picojson::value(sum);
    config["count"] = picojson::value(count);
    config["warmed_up"] = picojson::value(warmed_up);
    return config;
  }
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_METRIC_H_
