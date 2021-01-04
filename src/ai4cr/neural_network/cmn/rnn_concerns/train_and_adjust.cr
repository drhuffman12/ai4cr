require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainAndAdjust
          UNTIL_MIN_AVG_ERROR_DEFAULT = 0.1

          def train(input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
            eval(input_set_given)

            @output_set_expected = output_set_expected

            li_max = synaptic_layer_indexes_reversed.first
            ti_max = time_col_indexes_reversed.first

            synaptic_layer_indexes_reversed.each do |li|
              time_col_indexes_reversed.each do |ti|
                case
                when li == li_max && ti == ti_max
                  mini_net_set[li][ti].train(input_set_given[ti], @output_set_expected[ti], until_min_avg_error)
                when li == li_max && ti < ti_max
                  mini_net_set[li][ti].step_load_outputs(@output_set_expected[ti])

                  # step_backpropagate
                  step_calculate_output_deltas(li, ti)
                  mini_net_set[li][ti].step_calc_input_deltas
                  mini_net_set[li][ti].step_update_weights

                  mini_net_set[li][ti].step_calculate_error
                when li < li_max && ti == ti_max
                  mini_net_set[li][ti].step_load_outputs(hidden_outputs_expected(li, ti))

                  # step_backpropagate
                  step_calculate_output_deltas(li, ti)
                  mini_net_set[li][ti].step_calc_input_deltas
                  mini_net_set[li][ti].step_update_weights

                  # # step_calculate_error # N/A for li < li_max
                else
                  mini_net_set[li][ti].step_load_outputs(hidden_outputs_expected(li, ti))

                  # step_backpropagate
                  step_calculate_output_deltas(li, ti)
                  mini_net_set[li][ti].step_calc_input_deltas
                  mini_net_set[li][ti].step_update_weights

                  # # step_calculate_error # N/A for li < li_max
                end
              end
            end

            error_total
          end

          private def hidden_outputs_expected(li, ti)
            # Average of sums of (applicable) inputs_given and input_deltas from both forward directions
            # Only applicable for hidden outputs

            # raise "Only applicable for hidden outputs" if li == synaptic_layer_indexes_reversed.first

            outputs_expected = hidden_outputs_expected_from_next_li(li, ti)

            if ti < time_col_indexes_reversed.first
              hoefnti = hidden_outputs_expected_from_next_ti(li, ti)

              raise "oe discrepancy; li: #{li}, ti: #{ti}, hoefnti.size: #{hoefnti.size}, outputs_expected.size: #{outputs_expected.size}" if hoefnti.size != outputs_expected.size

              outputs_expected.map_with_index! do |oe, i|
                oe + hoefnti[i]
              end
            end

            outputs_expected
          end

          private def hidden_outputs_expected_from_next_li(li, ti)
            mini_net_set[li][ti].width_indexes.map do |i|
              mini_net_set[li + 1][ti].inputs_given[i] + mini_net_set[li + 1][ti].input_deltas[i]
            end
          end

          private def hidden_outputs_expected_from_next_ti(li, ti)
            from = node_input_sizes[li][ti + 1][:previous_synaptic_layer]
            to = from + mini_net_set[li][ti].width - 1

            (from..to).to_a.map do |i|
              mini_net_set[li][ti + 1].inputs_given[i] + mini_net_set[li][ti + 1].input_deltas[i]
            end
          end

          private def step_calculate_output_deltas(li, ti)
            slirf = synaptic_layer_indexes_reversed.first
            tcirf = time_col_indexes_reversed.first

            output_deltas = case
                            when li == slirf && ti == tcirf
                              # mini_net_set[li][ti].step_calculate_output_deltas
                              raise "Should never get here due to short-circuiting of logic"
                            when li == tcirf && ti < tcirf
                              ods = mini_net_set[li][ti].step_calculate_output_deltas
                              step_calculate_output_deltas_next_ti(li, ti, ods)
                            when li < tcirf && ti == tcirf
                              mini_net_set[li + 1][ti].input_deltas
                            else
                              ods = mini_net_set[li + 1][ti].input_deltas
                              step_calculate_output_deltas_next_ti(li, ti, ods)
                            end

            mini_net_set[li][ti].output_deltas = output_deltas
          end

          private def step_calculate_output_deltas_next_ti(li, ti, ods)
            from = node_input_sizes[li][ti + 1][:previous_synaptic_layer]
            to = from + mini_net_set[li][ti].width - 1

            # ods = mini_net_set[li][ti].step_calculate_output_deltas
            ods_next_ti = mini_net_set[li][ti + 1].input_deltas[from..to]
            ods.map_with_index do |od, i|
              od + ods_next_ti[i]
            end
          end
        end
      end
    end
  end
end
