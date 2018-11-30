# require "./math"
# require "./aliases"
require "./common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module NodeSet
        class Trainable < Common # Ai4cr::NeuralNetwork::Rnn::NodeSet::Trainable
          # include NodeSet::Common

          # MEMORY_QTY_DEFAULT = 1

          property state_errors : Array(Float64)
          property state_deltas : Array(Float64)
          property state_corrected_values : Array(Float64)

          # # memory aka history
          # getter memory_qty : Int32
          # getter memory_range : Range(Int32, Int32)

          # property memory_guesses : Array(Array(Float64))
          # property memory_errors : Array(Array(Float64))
          # property memory_deltas : Array(Array(Float64))

          def initialize(@state_qty = STATE_QTY_DEFAULT) # , @memory_qty = MEMORY_QTY_DEFAULT
            super(state_qty)

            # init_states
            @state_errors = init_state_values
            @state_deltas = init_state_values
            @state_corrected_values = init_state_values

            # # init_memories
            # @memory_range = (0..memory_qty-1)
            # @memory_guesses = init_memory_values
            # @memory_errors = init_memory_values
            # @memory_deltas = init_memory_values
          end

          # def init_memory_values
          #   memory_range.map{ |m| state_range.map{ |s| 0.0 } }
          # end
        end
      end
    end
  end
end
      