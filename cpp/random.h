/*!
 *  Copyright (c) 2023 by Contributors
 * \file random.h
 * \brief Header of random number generator.
 */

#ifndef MLC_LLM_RANDOM_H_
#define MLC_LLM_RANDOM_H_

#include <random>

namespace mlc {
namespace llm {

// Random number generator
class RandomGenerator {
 private:
  std::mt19937 gen;
  std::uniform_real_distribution<> dis;

 public:
  RandomGenerator(int seed = std::random_device{}()) : gen(seed), dis(0.0, 1.0) {}

  static RandomGenerator& GetInstance(int seed = std::random_device{}()) {
    static RandomGenerator instance(seed);
    return instance;
  }

  double GetRandomNumber() { return dis(gen); }

  void SetSeed(int seed) { gen.seed(seed); }
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_RANDOM_H_
