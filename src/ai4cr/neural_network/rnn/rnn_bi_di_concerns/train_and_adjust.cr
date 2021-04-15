# require "../rnn_simple_concerns/train_and_adjust.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnBiDiConcerns
        module TrainAndAdjust
          # include RnnSimpleConcerns::TrainAndAdjust
          # # This stays as-is (maybe):
          # private def step_calculate_output_error_along_ti(li, ti)
          #   raise "Index error" if ti == time_col_index_last

          #   from = node_input_sizes[li][ti + 1][:previous_synaptic_layer]
          #   to = from + mini_net_set[li][ti].width - 1

          #   mini_net_set[li][ti + 1].input_deltas[from..to]
          # end

          # # This gets added/implemented similar to 'step_calculate_output_error_along_ti', but in reverse time-col direction:
          # private def step_calculate_output_rev_error_along_ti(li, ti)
          #   # TODO

          #   # raise "Index error" if ti == time_col_index_last

          #   # from = node_input_sizes[li][ti + 1][:previous_synaptic_layer]
          #   # to = from + mini_net_set[li][ti].width - 1

          #   # mini_net_set[li][ti + 1].input_deltas[from..to]
          # end
        end
      end
    end
  end
end
