require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module CalcGuess
          EMPTY_1D_ARRAY_FLOAT64 = Array(Float64).new

          # steps for 'eval' aka 'guess':
          def eval(input_set_given)
            step_load_inputs(input_set_given)
            step_calc_forward

            outputs_guessed
          end

          def step_load_inputs(input_set_given)
            li = 0

            time_col_indexes.map do |ti|
              inputs = (input_set_given[ti] + step_load_previous_tc(li, ti)).flatten
              mini_net_set[li][ti].step_load_inputs(inputs)
            end
          end

          def step_calc_forward
            li = 0
            time_col_indexes.map do |ti|
              mini_net_set[li][ti].step_calc_forward
            end

            # Buggy ameba re Lint/ShadowingOuterLocalVar?
            # (Supposed to be fixed as per https://github.com/crystal-ameba/ameba/issues/147)
            # ameba:disable Lint/ShadowingOuterLocalVar
            synaptic_layer_indexes[1..-1].each do |li|
              time_col_indexes.map do |ti|
                inputs = (step_load_previous_li(li, ti) + step_load_previous_tc(li, ti)).flatten
                mini_net_set[li][ti].step_load_inputs(inputs)
                mini_net_set[li][ti].step_calc_forward
              end
            end
            # ameba:enable Lint/ShadowingOuterLocalVar
          end

          def all_mini_net_outputs
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].outputs_guessed
              end
            end
          end

          def outputs_guessed
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].outputs_guessed
            end
          end

          def guesses_sorted
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_sorted
            end
          end

          def guesses_sorted
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_sorted
            end
          end

          def guesses_ceiled
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_ceiled
            end
          end

          def guesses_top_n(n)
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_top_n(n)
            end
          end

          private def step_load_previous_tc(li, ti)
            ti > 0 ? mini_net_set[li][ti - 1].outputs_guessed : EMPTY_1D_ARRAY_FLOAT64
          end

          private def step_load_previous_li(li, ti)
            li > 0 ? mini_net_set[li - 1][ti].outputs_guessed : EMPTY_1D_ARRAY_FLOAT64
          end
        end
      end
    end
  end
end
