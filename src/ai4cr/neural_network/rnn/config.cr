require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct Config
        include JSON::Serializable
        # include Common::Initializers::LearningRate
        # include Common::Initializers::Momentum
        include Ai4cr::NeuralNetwork::Backpropagation::Common::Initializers::LearningRate
        include Ai4cr::NeuralNetwork::Backpropagation::Common::Initializers::Momentum

        property qty_states_in
        property qty_states_out
        property qty_time_cols
        property qty_lpfc_layers
        property qty_recent_memory

        property structure_hidden_laters : Array(Int32)
        property disable_bias : Bool

        def initialize(
          # RNN Net:
          @qty_states_in = 3, @qty_states_out = 4, @qty_time_cols = 5, @qty_lpfc_layers = 2, @qty_recent_memory = 2,
          # Embedded Backpropagation Nets:
          @structure_hidden_laters = [] of Int32, disable_bias = true, learning_rate = nil, momentum = nil
        )
          @disable_bias = !!disable_bias
          @learning_rate = init_learning_rate(learning_rate)
          @momentum = init_momentum(momentum)
        end
      end
    end
  end
end

# touch spec/ai4cr/neural_network/rnn/config_spec.cr