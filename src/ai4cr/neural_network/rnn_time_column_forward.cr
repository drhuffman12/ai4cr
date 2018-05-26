# Ai4cr::NeuralNetwork::RnnTimeColumnForward
module Ai4cr
  module NeuralNetwork
    struct RnnTimeColumnForward
        include Ai4cr::NeuralNetwork::Concerns::Backprop
        include Ai4cr::NeuralNetwork::Concerns::RtcFrwd
    end
  end
end
