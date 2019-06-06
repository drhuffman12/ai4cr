require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct Config
        include JSON::Serializable
        include NeuralNetwork::Common::Initializers::StructureHiddenLaters
        include NeuralNetwork::Common::Initializers::LearningRate
        include NeuralNetwork::Common::Initializers::Momentum

        property qty_states_in
        property qty_states_hidden_out
        property qty_states_out
        property qty_time_cols
        property qty_lpfc_layers
        property qty_hidden_laters
        property qty_time_cols_neighbor_inputs
        property qty_recent_memory

        # property structure_hidden_laters : Array(Int32)
        # property disable_bias : Bool

        def initialize(
          # RNN Net:
          @qty_states_in = 3, @qty_states_hidden_out = 5, @qty_states_out = 4,
          @qty_time_cols = 5,
          @qty_lpfc_layers = 3, @qty_hidden_laters = 2,
          @qty_time_cols_neighbor_inputs = 2, @qty_recent_memory = 2,
          # Embedded Backpropagation Nets:
          # @structure_hidden_laters = [] of Int32,
          # @hidden_scale_factor = 2,
          # structure_hidden_laters : Array(Int32)? = nil,
          # disable_bias = true, 
          learning_rate = nil, momentum = nil
        )
          @structure_hidden_laters = init_structure_hidden_laters(qty_states_hidden_out, qty_hidden_laters)
          # @disable_bias = !!disable_bias
          @learning_rate = init_learning_rate(learning_rate)
          @momentum = init_momentum(momentum)
        end
      end
    end
  end
end
