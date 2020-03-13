require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module CalcGuess
          def eval(input_sets_given)
            @layer_range.map do |h|
              @time_col_range.map do |t|
                net = @mini_net_set[h][t]

                # A) collect inputs:
                inputs_all = Array(Array(Float64)).new
                # add prev layer inputs
                if h == 0
                  inputs_all << input_sets_given[t]
                else
                  inputs_all << @mini_net_set[h - 1][t].outputs_guessed
                end

                # add prev time col inputs
                inputs_all << @mini_net_set[h][t - 1].outputs_guessed if t > 0

                # B) do calc's:
                net.step_load_inputs(inputs_all.flatten)
                net.step_calc_forward
              end
            end

            outputs_guessed
          end

          # def eval(input_s_given)
          #   @mini_net_set.each_with_index do |net, index|
          #     # index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)

          #     # load inputs
          #     if index == 0
          #       net.step_load_inputs(inputs_given)
          #     else
          #       net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)
          #     end

          #     net.step_calc_forward
          #   end

          #   @mini_net_set.last.outputs_guessed
          # end

          def outputs_guessed
            h = @layer_range.last
            @outputs_guessed = @time_col_range.map do |t|
              @mini_net_set[h][t].outputs_guessed
            end
          end

          def guesses_best
            h = @layer_range.last
            @time_col_range.map do |t|
              @mini_net_set[h][t].guesses_best
            end
          end

          # def step_calculate_error_distance_history
          #   @mini_net_set.last.step_calculate_error_distance_history
          # end

          # def error_distance_history
          #   @mini_net_set.last.error_distance_history
          # end
        end
      end
    end
  end
end
