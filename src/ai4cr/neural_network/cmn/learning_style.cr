module Ai4cr
  module NeuralNetwork
    module Cmn
      # module MiniNet
      #   module MiniNetConcerns
      # Ai4cr::NeuralNetwork::Cmn::LearningStyle.values
      # => [Prelu, Relu, Sigmoid, Tanh]
      #
      # Ai4cr::NeuralNetwork::Cmn::LearningStyle.values.map{|v|v.to_s}
      # => ["Prelu", "Relu", "Sigmoid", "Tanh"]
      #
      # Ai4cr::NeuralNetwork::Cmn::LearningStyle.values.map{|v|v.value}
      # => [10, 11, 12, 13]
      #
      # Ai4cr::NeuralNetwork::Cmn::LearningStyle::Prelu.value
      # => 10
      #
      # Ai4cr::NeuralNetwork::Cmn::LS_PRELU.value
      # => 10
      #
      enum LearningStyle
        Prelu   = 10
        Relu    = 20
        Sigmoid = 30
        Tanh    = 40
      end
      #   end
      # end

      # Ai4cr::NeuralNetwork::Cmn::
      LS_PRELU   = LearningStyle::Prelu
      LS_RELU    = LearningStyle::Relu
      LS_SIGMOID = LearningStyle::Sigmoid
      LS_TANH    = LearningStyle::Tanh
    end
  end
end
