require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainAndAdjust
          def train(input_set_given, output_set_expected, until_min_avg_error = 0.1)
            # guess forward:
            step_load_inputs(input_set_given)
            step_calc_forward

            # train backwards:
            step_load_outputs(output_set_expected)
            step_calculate_error

            step_backpropagate

            # resulting error:
            @error_total
          end

          def step_load_outputs(output_set_expected)
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              outputs = output_set_expected[ti]
              mini_net_set[li][ti].step_load_outputs(outputs)
            end
          end

          def step_calculate_error
            li = synaptic_layer_indexes.last

            @error_per_ti = time_col_indexes.map do |ti|
              mini_net_set[li][ti].step_calculate_error
            end

            @error_total = @error_per_ti.sum
          end

          def step_backpropagate
            synaptic_layer_indexes_reversed.each do |li|
              time_col_indexes_reversed.map do |ti|
                # TODO: Should I be collecting 'input_deltas' or sum of 'inputs' and 'input_deltas'?

                if li == synaptic_layer_indexes_reversed.first && ti == time_col_indexes_reversed.first
                  mini_net_set[li][ti].step_backpropagate
                else
                  # step_calculate_output_deltas

                  ods = mini_net_set[li][ti].step_calculate_output_deltas
                  id_nli = step_input_deltas_from_next_li(li, ti)
                  id_nti = step_input_deltas_from_next_tc(li, ti)

                  puts "----"
                  puts "BEFORE:: mini_net_set[li][ti].output_deltas: #{mini_net_set[li][ti].output_deltas}"
                  puts "VS"

                  mini_net_set[li][ti].output_deltas = ods.map_with_index do |od, i|
                    # od + id_nli&.[](i) + id_nti&.[](i)
                    # od + id_nli.try(&.[i]) + id_nti.try(&.[i])
                    o = od
                    o += id_nli[i] if id_nli.size > 0
                    o += id_nti[i] if id_nti.size > 0
                    o
                  end

                  puts "AFTER:: mini_net_set[li][ti].output_deltas: #{mini_net_set[li][ti].output_deltas}"
                  puts "----"

                  # step_calc_input_deltas
                  mini_net_set[li][ti].step_calc_input_deltas

                  # step_update_weights
                  mini_net_set[li][ti].step_update_weights
                end
              end
            end
          end

          private def step_input_deltas_from_next_tc(li, ti)
            if ti < time_col_indexes.last
              nis = node_input_sizes[li][ti + 1]
              psl = nis[:previous_synaptic_layer]
              ptc = nis[:previous_time_column]
              # input_deltas = mini_net_set[li][ti + 1].input_deltas[0..psl-1]
              mini_net_set[li][ti + 1].input_deltas[psl..psl + ptc - 1]
            else
              # Should never get called!?
              # EMPTY_1D_ARRAY_FLOAT64
              Array(Float64).new
            end
          end

          private def step_input_deltas_from_next_li(li, ti)
            if li < synaptic_layer_indexes.last
              nis = node_input_sizes[li + 1][ti]
              psl = nis[:previous_synaptic_layer]
              # ptc = nis[:previous_time_column]
              mini_net_set[li + 1][ti].input_deltas[0..psl - 1]
              # input_deltas = mini_net_set[li][ti + 1].input_deltas[psl..psl+ptc-1]
            else
              # Should never get called!?
              # EMPTY_1D_ARRAY_FLOAT64
              Array(Float64).new
            end
          end
        end
      end
    end
  end
end
