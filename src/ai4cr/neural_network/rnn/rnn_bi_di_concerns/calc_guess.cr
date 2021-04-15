module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnBiDiConcerns
        module CalcGuess
          # include RnnSimpleConcerns::CalcGuess
          # This stays as-is:
          # property node_input_sizes = Array(Array(NamedTuple(
          #   previous_synaptic_layer: Int32,
          #   previous_time_column: Int32
          # ))).new

          # # This gets added/implemented similar to 'node_input_sizes', but in reverse time-col direction:
          # property node_rev_input_sizes = Array(Array(NamedTuple(
          #   previous_synaptic_layer: Int32,
          #   next_time_column: Int32
          # ))).new

          # property node_near_input_sizes = Array(Array(NamedTuple(
          #   previous_synaptic_layer: Int32,
          #   previous_time_column: Int32,
          #   next_time_column: Int32
          # ))).new

        end
      end
    end
  end
end
