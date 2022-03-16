/*
 * FASHION.h
 *
 */

#ifndef FASHION_H_
#define FASHION_H_

#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/script.h>
#include <torch/torch.h>

//#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <string>


/// The FASHION dataset.

class FASHION : public torch::data::datasets::Dataset<FASHION>  {
 public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  // Loads the CIFAR10 dataset from the `root` path.
      //
      // The supplied `root` path should contain the *content* of the unzipped
      explicit FASHION(const std::string& root, Mode mode = Mode::kTrain);

      // Returns the `Example` at the given `index`.
      torch::data::Example<> get(size_t index) override;

      // Returns the size of the dataset.
      torch::optional<size_t> size() const override;

      // Returns true if this is the training subset of CIFAR10.
      bool is_train() const noexcept;

      // Returns all images stacked into a single tensor.
      const torch::Tensor& images() const;

      // Returns all targets stacked into a single tensor.
      const torch::Tensor& targets() const;

   private:
      torch::Tensor images_;
      torch::Tensor targets_;
      Mode mode_;
};

#endif /* FASHION_H_ */
