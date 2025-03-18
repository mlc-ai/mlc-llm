/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/random.h
 * \brief Header of random number generator.
 */

#ifndef MLC_LLM_SUPPORT_RANDOM_H_
#define MLC_LLM_SUPPORT_RANDOM_H_

#include <random>
#include <stdexcept>

namespace mlc {
namespace llm {

// Base class for random number generators.
class RandomGenerator {
 private:
  int seed_;

 public:
  RandomGenerator(int seed = std::random_device{}()) : seed_(seed) {}

  static RandomGenerator& GetInstance(int seed = std::random_device{}()) {
    static RandomGenerator instance(seed);
    return instance;
  }

  // Returns a random number in [0, 1).
  virtual double GetRandomNumber() {
    throw std::runtime_error("GetRandomNumber() not implemented");
  }

  // Returns a Philox offset based on the increment.
  virtual uint64_t GetPhiloxOffset(uint64_t increment) {
    throw std::runtime_error("GetPhiloxOffset() not implemented");
  }

  // Retrieves the seed.
  int GetSeed() const { return seed_; }
};

class UniformRandomGenerator : public RandomGenerator {
 private:
  std::mt19937 gen;
  std::uniform_real_distribution<> dis;

 public:
  UniformRandomGenerator(int seed = std::random_device{}())
      : RandomGenerator(seed), gen(seed), dis(0.0, 1.0) {}

  double GetRandomNumber() override { return dis(gen); }
};

// Primarily for state tracking
class PhiloxRandomGenerator : public RandomGenerator {
 private:
  uint64_t offset_;

 public:
  PhiloxRandomGenerator(int seed = std::random_device{}()) : RandomGenerator(seed), offset_(0) {}

  uint64_t GetPhiloxOffset(uint64_t increment) override {
    offset_ += increment;
    return offset_;
  }
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_RANDOM_H_
