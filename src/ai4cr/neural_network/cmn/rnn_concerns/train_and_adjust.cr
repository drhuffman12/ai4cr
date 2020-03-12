require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainAndAdjust
          def train(input_sets_given, output_sets_expected) # , until_min_avg_error = 0.1)
          end

          # # TODO: utilize until_min_avg_error

          # def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
          #   @mini_net_set.each_with_index do |net, index|
          #     index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)
          #     net.step_calc_forward
          #   end

          #   index_max = @mini_net_set.size - 1
          #   (0..index_max).to_a.reverse.each do |index|
          #     net = @mini_net_set[index]

          #     # index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@mini_net_set[index + 1].input_deltas[0..@mini_net_set[index + 1].height - 1])
          #     index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@mini_net_set[index + 1].input_deltas)

          #     net.step_calculate_error
          #     net.step_backpropagate
          #   end

          #   @mini_net_set.last.error_total
          # # end
        end
      end
    end
  end
end
