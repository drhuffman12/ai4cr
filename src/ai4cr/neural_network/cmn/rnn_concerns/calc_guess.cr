require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module CalcGuess
          # steps for 'eval' aka 'guess':
          def eval(input_set_given)
            # TODO: Review/compare w/ 'train' and adjust as applicable!

            @input_set_given = input_set_given

            synaptic_layer_indexes.each do |li|
              time_col_indexes.each do |ti|
                mini_net_set[li][ti].step_load_inputs(inputs_for(li, ti))
                mini_net_set[li][ti].step_calc_forward
              end
            end

            outputs_guessed
          end

          def inputs_for(li, ti)
            case
            when li == 0 && ti == 0
              @input_set_given[ti]
            when li == 0 && ti > 0
              @input_set_given[ti] + step_outputs_guessed_from_previous_tc(li, ti)
            when li > 0 && ti == 0
              step_outputs_guessed_from_previous_li(li, ti)
            else
              step_outputs_guessed_from_previous_li(li, ti) + step_outputs_guessed_from_previous_tc(li, ti)
            end
          end

          def outputs_guessed
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].outputs_guessed
            end
          end

          private def step_outputs_guessed_from_previous_tc(li, ti)
            # ti > 0 ? mini_net_set[li][ti - 1].outputs_guessed : Array(Float64).new
            raise "Index error" if ti == 0

            mini_net_set[li][ti - 1].outputs_guessed
          end

          private def step_outputs_guessed_from_previous_li(li, ti)
            # li > 0 ? mini_net_set[li - 1][ti].outputs_guessed : Array(Float64).new
            raise "Index error" if li == 0

            mini_net_set[li - 1][ti].outputs_guessed
          end

          # guesses
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
        end
      end
    end
  end
end
