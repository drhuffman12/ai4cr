# require "./math"
# require "./aliases"
require "./trainable"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module NodeSet
        class Hidden < Trainable # Ai4cr::NeuralNetwork::Rnn::NodeSet::Common
          MEMORY_QTY_DEFAULT = 2

          # property state_errors : Array(Float64)
          # property state_deltas : Array(Float64)

          # memory aka history
          getter memory_qty : Int32
          getter memory_range : Range(Int32, Int32)

          property memory_values_set : Array(Array(Float64))
          property memory_errors_set : Array(Array(Float64))
          property memory_deltas_set : Array(Array(Float64))
          property memory_corrected_values_set : Array(Array(Float64))

          def initialize(@state_qty = STATE_QTY_DEFAULT, @memory_qty = MEMORY_QTY_DEFAULT)
            super(state_qty)

            # init_memories
            @memory_range = (0..memory_qty-1)
            @memory_values_set = init_memory_values_set
            @memory_errors_set = init_memory_values_set
            @memory_deltas_set = init_memory_values_set
            @memory_corrected_values_set = init_memory_values_set
          end

          def init_memory_values_set
            memory_range.map{ |m| state_range.map{ |s| 0.0 } }
          end
        end
      end
    end
  end
end
      