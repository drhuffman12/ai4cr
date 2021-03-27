module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleConcerns
        module TrainAndAdjust
          # TODO: Finish refactoring this and 'Ai4cr::Team' and then do code cleanup!

          getter output_set_expected = Array(Array(Float64)).new
          getter all_output_errors = Array(Array(Float64)).new

          # def eval_and_compare(input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
          #   eval(input_set_given)
          #   @output_set_expected = output_set_expected
          #   li = synaptic_layer_index_last
          #   time_col_indexes_reversed.each do |ti|
          #     mini_net_set[li][ti].step_load_outputs(@output_set_expected[ti])
          #     step_calculate_output_deltas(li, ti)
          #     mini_net_set[li][ti].calculate_error_distance
          #   end
          #   # error_stats.distance
          #   calculate_error_distance
          # end

          def train(input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
            # TODO: WHY is the last ti's value in outputs_guessed reset to '0.0' after training (but NOT after eval'ing)??? (and NOT reset to '0.0' after next round of training???)

            eval(input_set_given)
            @output_set_expected = output_set_expected

            synaptic_layer_indexes_reversed.each do |li|
              time_col_indexes_reversed.each do |ti|
                case
                when li == synaptic_layer_index_last && ti == time_col_index_last
                  # In this case, to calculate the 'outputs_expected', we just use @output_set_expected[ti].
                  # 'regular' for all sub-parts
                  # e.g.:
                  # * 'regular' output_deltas
                  mini_net_set[li][ti].train(input_set_given[ti], @output_set_expected[ti], until_min_avg_error)
                when li == synaptic_layer_index_last && ti < time_col_index_last
                  # This is an odd situation.
                  # In this case, to calculate the 'outputs_expected', we just use @output_set_expected[ti], which must 'rule' the 'outputs_expected' values.
                  # However, we also need to consider error adjustments coming from the next time column index.
                  # We want to combine the error deltas coming from both li and ti directions,
                  #   but we must first load the 'output_set_expected' values to be able to calculate the deltas along the li axis.
                  mini_net_set[li][ti].step_load_outputs(@output_set_expected[ti]) # 'regular' step_load_outputs
                  step_calculate_output_errors_at(li, ti)
                  step_backpropagate(li, ti)
                when li < synaptic_layer_index_last && ti == time_col_index_last
                  # In this case, to calculate the 'outputs_expected', use outputs_guessed (of current [li][ti]) + input_deltas of (matching parts of next [li][ti])
                  #   TODO: Should this be plus or minus?
                  # Also, this is calc'd only in the li direction.
                  step_calculate_output_errors_at(li, ti)
                  mini_net_set[li][ti].step_load_outputs(calc_hidden_outputs_expected(li, ti))
                  step_backpropagate(li, ti)
                when li < synaptic_layer_index_last && ti < time_col_index_last
                  # In this case, to calculate the 'outputs_expected', use outputs_guessed (of current [li][ti]) + input_deltas of (matching parts of next [li][ti])
                  #   TODO: Should this be plus or minus?
                  # Also, this is calc'd as a combo (sum or avg) in both li and ti directions.
                  step_calculate_output_errors_at(li, ti)
                  mini_net_set[li][ti].step_load_outputs(calc_hidden_outputs_expected(li, ti))
                  step_backpropagate(li, ti)
                else
                  raise "Index error! (Range Mis-match!) li: #{li}, ti: #{ti}"
                end
              end
            end

            calculate_error_distance
          end

          private def step_backpropagate(li, ti)
            step_calculate_output_deltas(li, ti)
            mini_net_set[li][ti].step_calc_input_deltas
            mini_net_set[li][ti].step_update_weights
            # mini_net_set[li][ti].auto_shrink_weights
            mini_net_set[li][ti].calculate_error_distance
          end

          private def step_calculate_output_errors_at(li, ti)
            mns = mini_net_set[li][ti]

            local_errors = case
                           when li == synaptic_layer_index_last && ti == time_col_index_last
                             raise "Index Error! Invalid method!"
                           when li == synaptic_layer_index_last && ti < time_col_index_last
                             # We have 2 errors to deal with; we will average them.
                             error_along_li = step_calculate_output_error_along_li(li, ti)
                             error_along_ti = step_calculate_output_error_along_ti(li, ti)
                             error_along_li.map_with_index { |eli, i| 0.5 * (eli + error_along_ti[i]) }
                           when li < synaptic_layer_index_last && ti == time_col_index_last
                             step_calculate_output_error_along_li(li, ti)
                           when li < synaptic_layer_index_last && ti < time_col_index_last
                             # We have 2 errors to deal with; we will average them.
                             error_along_li = step_calculate_output_error_along_li(li, ti)
                             error_along_ti = step_calculate_output_error_along_ti(li, ti)
                             error_along_li.map_with_index { |eli, i| 0.5 * (eli + error_along_ti[i]) }
                           else
                             raise "Index error! (Range Mis-match!) li: #{li}, ti: #{ti}"
                           end

            mns.output_errors = local_errors
          end

          private def step_calculate_output_error_along_li(li, ti)
            if li == synaptic_layer_index_last # && ti < time_col_index_last
              mini_net_set[li][ti].step_calc_output_errors
            else
              from = 0
              to = mini_net_set[li][ti].width - 1
              mini_net_set[li + 1][ti].input_deltas[from..to]
            end
          end

          private def step_calculate_output_error_along_ti(li, ti)
            raise "Index error" if ti == time_col_index_last

            from = node_input_sizes[li][ti + 1][:previous_synaptic_layer]
            to = from + mini_net_set[li][ti].width - 1

            mini_net_set[li][ti + 1].input_deltas[from..to]
          end

          private def calc_hidden_outputs_expected(li, ti) # TODO: review and revise?
            # This is ONLY valid AFTER 'step_calculate_output_errors_at' is called!!!!
            raise "Index Error" if li == synaptic_layer_index_last

            og = mini_net_set[li][ti].outputs_guessed.clone
            oe = mini_net_set[li][ti].output_errors.clone
            og.map_with_index { |o, i| o + oe[i] }
          end

          private def step_calculate_output_deltas(li, ti)
            # NOTE: We must use a modified logic compared to MiniNet, which is:
            # ```
            # @output_deltas.map_with_index! do |_, i|
            #   error = @outputs_expected[i] - @outputs_guessed[i]
            #   derivative_propagation_function.call(@outputs_guessed[i]) * error
            # end
            # ```

            mns = mini_net_set[li][ti]
            mns.output_deltas.map_with_index! do |_, i|
              mns.derivative_propagation_function.call(mns.outputs_guessed[i].clone) * mns.output_errors[i].clone
            end
          end

          def final_li_output_error_distances
            li = synaptic_layer_indexes.last
            time_col_indexes.map do |ti|
              mini_net_set[li][ti].error_stats.distance
            end
          end

          def calculate_error_distance
            @error_stats.distance = final_li_output_error_distances.sum { |e| 0.5*(e)**2 }
          end
        end
      end
    end
  end
end
