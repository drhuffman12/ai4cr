require "json"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct Net
        include JSON::Serializable

        property state : Rnn::State

        def initialize(
          # RNN Net:
          qty_states_in = 3, qty_states_hidden_out = 5, qty_states_out = 4,
          qty_time_cols = 5,
          qty_lpfc_layers = 3,
          qty_hidden_laters = 2,
          qty_time_cols_neighbor_inputs = 2,
          qty_recent_memory = 2,
          
          # Embedded Backpropagation Nets:
          learning_rate = nil, momentum = nil
        )
          @state = Rnn::State.new(
            # RNN Net:
            qty_states_in = 3, qty_states_hidden_out = 5, qty_states_out = 4,
            qty_time_cols = 5,
            qty_lpfc_layers = 3,
            qty_hidden_laters = 2,
            qty_time_cols_neighbor_inputs = 2,
            qty_recent_memory = 2,
            
            # Embedded Backpropagation Nets:
            learning_rate = nil, momentum = nil
          )
        end

        # def collect_inputs
        #   # @inputs = ...
        # end

        # # def inputs
        # #   @inputs = ...
        # # end

        # def guess_forward # (_inputs)
        # end

        # def train_backwards # (_outputs)
        # end

        # def outputs
        # end
      end
    end
  end
end
      