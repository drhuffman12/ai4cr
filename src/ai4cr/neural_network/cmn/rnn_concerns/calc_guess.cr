require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module CalcGuess
          def eval(input_sets_given)
          end

          # def eval(input_s_given)
          #   @net_set.each_with_index do |net, index|
          #     # index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@net_set[index - 1].outputs_guessed)

          #     # load inputs
          #     if index == 0
          #       net.step_load_inputs(inputs_given)
          #     else
          #       net.step_load_inputs(@net_set[index - 1].outputs_guessed)
          #     end

          #     net.step_calc_forward
          #   end

          #   @net_set.last.outputs_guessed
          # end

          # def guesses_best
          #   @net_set.last.guesses_best
          # end

          # def step_calculate_error_distance_history
          #   @net_set.last.step_calculate_error_distance_history
          # end

          # def error_distance_history
          #   @net_set.last.error_distance_history
          # end
        end
      end
    end
  end
end
