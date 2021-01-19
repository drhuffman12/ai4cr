module Ai4cr
  module NeuralNetwork
    enum LearningStyle
      Prelu   = 10
      Relu    = 20
      Sigmoid = 30
      Tanh    = 40
    end

    # Ai4cr::NeuralNetwork::
    LS_PRELU   = LearningStyle::Prelu
    LS_RELU    = LearningStyle::Relu
    LS_SIGMOID = LearningStyle::Sigmoid
    LS_TANH    = LearningStyle::Tanh
  end
end
