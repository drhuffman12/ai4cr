module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module TaaDistinct
            def train(input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
              # TODO: WHY is the last tci's value in outputs_guessed reset to '0.0' after training (but NOT after eval'ing)??? (and NOT reset to '0.0' after next round of training???)

              eval(input_set_given)
              @output_set_expected = output_set_expected

              # all_mini_nets_each_reversed_with_indexes do |mininet, sli, tci, channel|
              #   case
              #   when sli == synaptic_layer_indexes.last && channel == :channel_sl_or_combo
              #     # step_load_outputs
              #     mininet.step_load_outputs(@output_set_expected[tci])
              #     # step_calculate_output_errors_at
              #     step_calculate_output_errors_at(sli, tci, channel, mininet)
              #     # step_backpropagate
              #     step_backpropagate_at(sli, tci, channel, mininet)
              #   else
              #   end
              # end

              synaptic_layer_indexes_reversed.each do |sli|
                train_channel_sl_or_combo(sli)
                if sli > 0
                  train_channel_backwards(sli)
                  train_channel_forwards(sli)
                end
              end

              calculate_error_distance
            end

            def train_channel_sl_or_combo(sli)
              time_col_indexes_reversed.each do |tci|
                outputs_expected = outputs_expected_for(sli, tci, :channel_sl_or_combo)
                mini_net_set[sli][tci][:channel_sl_or_combo].step_load_outputs(outputs_expected)
                mini_net_set[sli][tci][:channel_sl_or_combo].step_calc_output_errors
                # step_calculate_output_errors_at(sli, tci, :channel_sl_or_combo)

                # step_backpropagate_at(sli, tci, :channel_sl_or_combo)
                mini_net_set[sli][tci][:channel_sl_or_combo].step_backpropagate
                # mini_net_set[sli][tci][:channel_sl_or_combo].step_calculate_output_deltas
                # mini_net_set[sli][tci][:channel_sl_or_combo].step_calc_input_deltas
                # mini_net_set[sli][tci][:channel_sl_or_combo].step_update_weights
              end
            end

            def train_channel_backwards(sli)
              # @input_set_given
              # @output_set_expected
              time_col_indexes.each do |tci|
                outputs_expected = outputs_expected_for(sli, tci, :channel_backwards)
                mini_net_set[sli][tci][:channel_backwards].step_load_outputs(outputs_expected)
                mini_net_set[sli][tci][:channel_backwards].step_calc_output_errors
                # step_calculate_output_errors_at(sli, tci, :channel_backwards)

                # step_backpropagate_at(sli, tci, :channel_backwards)
                mini_net_set[sli][tci][:channel_backwards].step_backpropagate
                # mini_net_set[sli][tci][:channel_backwards].step_calculate_output_deltas
                # mini_net_set[sli][tci][:channel_backwards].step_calc_input_deltas
                # mini_net_set[sli][tci][:channel_backwards].step_update_weights
              end
            end

            def train_channel_forwards(sli)
              # @input_set_given
              # @output_set_expected
              time_col_indexes_reversed.each do |tci|
                outputs_expected = outputs_expected_for(sli, tci, :channel_forwards)
                mini_net_set[sli][tci][:channel_forwards].step_load_outputs(outputs_expected)
                mini_net_set[sli][tci][:channel_forwards].step_calc_output_errors
                # step_calculate_output_errors_at(sli, tci, :channel_forwards)

                # step_backpropagate_at(sli, tci, :channel_forwards)
                mini_net_set[sli][tci][:channel_forwards].step_backpropagate
                # mini_net_set[sli][tci][:channel_forwards].step_calculate_output_deltas
                # mini_net_set[sli][tci][:channel_forwards].step_calc_input_deltas
                # mini_net_set[sli][tci][:channel_forwards].step_update_weights
              end
            end

            # ameba:disable Metrics/CyclomaticComplexity
            def outputs_expected_for(sli, tci, channel)
              # TODO: Split this method up!

              # @input_set_given
              # @output_set_expected

              # The 'outs_expected' is only (mostly) directly defined by outputs_expected for
              #   th last sli of last tci of channel ':channel_sl_or_combo'.
              #   But that has 'memory' also, which must be considered.
              # All others must first gather applicable 'outs_deltas' and derive applicable combo of 'outs_expected'.
              outs_expected = Array(Float64).new
              outs_deltas = Hash(Symbol, Array(Float64)).new

              # outs_deltas[:current_self_mem] (all have it; except on first sli, only channel channel_sl_or_combo exists)
              if channel == :channel_sl_or_combo || sli > 0
                outs_deltas[:current_self_mem] = outs_deltas_mem(sli, tci, channel)
              end

              # ### TODO... (left off here as of 2021-05-38)

              case
              when sli == synaptic_layer_indexes.last && channel == :channel_sl_or_combo
                # final outputs

                puts
                puts "v"*80
                p! sli
                p! tci
                p! @output_set_expected
                puts "^"*80
                puts

                outs_expected = @output_set_expected[tci]
              when sli > 0 && tci == 0 && channel == :channel_backward
                # channel_forward
                # (none)

                # channel_backward
                # (none)

                # channel_sl_or_combo
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli][tci][channel_from]
                nis_from = node_input_sizes[sli][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo] # +
                #   nis_from[:current_forward]
                # i_to = i_from + nis_from[:current_backward] - 1
                # outs_deltas[:sl_next_combo] = mn_from.input_deltas[i_from..i_to]

              when sli > 0 && channel == :channel_backward
                # channel_forward
                # (none)

                # channel_backward
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli][tci - 1][channel_from]
                nis_from = node_input_sizes[sli][tci - 1][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo] +
                #   nis_from[:sl_previous_channel_backward]
                # i_to = i_from + nis_from[:tc_next_channel_backward] - 1
                # outs_deltas[:tc_previous_channel_backward] = mn_from.input_deltas[i_from..i_to]

                # channel_sl_or_combo
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli][tci][channel_from]
                nis_from = node_input_sizes[sli][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo] +
                #   nis_from[:current_forward]
                # i_to = i_from + nis_from[:current_backward] - 1
                # outs_deltas[:sl_next_combo] = mn_from.input_deltas[i_from..i_to]

              when sli > 0 && tci == time_col_indexes_last && channel == :channel_forward
                # channel_forward
                # (none)

                # channel_backward
                # (none)

                # channel_sl_or_combo
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli][tci][channel_from]
                nis_from = node_input_sizes[sli][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo]
                # i_to = i_from + nis_from[:current_forward] - 1
                # outs_deltas[:sl_next_combo] = mn_from.input_deltas[i_from..i_to]

              when sli > 0 && channel == :channel_forward
                # channel_forward
                channel_from = :channel_forward
                mn_from = mini_net_set[sli][tci + 1][channel_from]
                nis_from = node_input_sizes[sli][tci + 1][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo]
                # i_to = i_from + nis_from[:sl_previous_channel_forward] - 1
                # outs_deltas[:tc_next_channel_forward] = mn_from.input_deltas[i_from..i_to]

                # channel_backward
                # (none)

                # channel_sl_or_combo
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli][tci][channel_from]
                nis_from = node_input_sizes[sli][tci][channel_from]

                puts
                puts "v"*80
                puts "Error: missing key 'current_forward' for named tuple NamedTuple(current_self_mem: Int32, sl_previous_input_or_combo: Int32, sl_previous_channel_backward: Int32, tc_next_channel_backward: Int32)"
                p! sli
                p! tci
                p! channel
                p! channel_from
                p! mn_from
                p! nis_from
                puts "^"*80
                puts

                # i_from = nis_from[:current_self_mem] +
                #   nis_from[:sl_previous_input_or_combo]
                # i_to = i_from + nis_from[:current_forward] - 1
                # outs_deltas[:sl_next_combo] = mn_from.input_deltas[i_from..i_to]

              when channel == :channel_sl_or_combo
                # channel_forward
                channel_from = :channel_forward
                mn_from = mini_net_set[sli + 1][tci][channel_from]
                nis_from = node_input_sizes[sli + 1][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel_from
                p! nis_from
                puts "^"*80
                puts

                i_from = nis_from[:current_self_mem]
                i_to = i_from + nis_from[:sl_previous_input_or_combo] - 1
                outs_deltas[:sl_next_channel_forward] = mn_from.input_deltas[i_from..i_to]

                # channel_backward
                channel_from = :channel_backward
                mn_from = mini_net_set[sli + 1][tci][channel_from]
                nis_from = node_input_sizes[sli + 1][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel_from
                p! nis_from
                puts "^"*80
                puts

                i_from = nis_from[:current_self_mem]
                i_to = i_from + nis_from[:sl_previous_input_or_combo] - 1
                outs_deltas[:sl_next_channel_backward] = mn_from.input_deltas[i_from..i_to]

                # channel_sl_or_combo
                channel_from = :channel_sl_or_combo
                mn_from = mini_net_set[sli + 1][tci][channel_from]
                nis_from = node_input_sizes[sli + 1][tci][channel_from]

                puts
                puts "v"*80
                p! sli
                p! tci
                p! channel_from
                p! nis_from
                puts "^"*80
                puts

                i_from = nis_from[:current_self_mem]
                i_to = i_from + nis_from[:sl_previous_input_or_combo] - 1
                outs_deltas[:sl_next_channel_combo] = mn_from.input_deltas[i_from..i_to]

                # else
                #   raise "Other combo's should not exist!"
              end

              {outs_deltas: outs_deltas, outs_expected: outs_expected}
            end

            # ameba:enable Metrics/CyclomaticComplexity

            def outs_deltas_mem(sli, tci, channel)
              # TODO: How to calc input_deltas for inclusion in calcs for gathering outputs expected and/or deltas?
              #   i.e.: The below will use a 1-training-round offset for memory; kinda like a blurred-memory update?
              #         But, I think that's ok, since the first training round has memory inputs of all 0's.
              channel_from = channel
              mn_from = mini_net_set[sli][tci][channel_from]
              nis_from = node_input_sizes[sli][tci][channel_from]
              i_from = 0
              i_to = nis_from[:current_self_mem] - 1
              mn_from.input_deltas[i_from..i_to]
            end

            # private def step_backpropagate(sli, tci)
            #   step_calculate_output_deltas(sli, tci)
            #   mini_net_set[sli][tci].step_calc_input_deltas
            #   mini_net_set[sli][tci].step_update_weights
            #   mini_net_set[sli][tci].calculate_error_distance
            # end

            # private def step_calculate_output_errors_at(sli, tci)
            #   mns = mini_net_set[sli][tci]

            #   local_errors = case
            #                  when sli == synaptic_layer_index_last && tci == time_col_index_last
            #                    raise "Index Error! Invalid method!"
            #                  when sli == synaptic_layer_index_last && tci < time_col_index_last
            #                    # We have 2 errors to deal with; we will average them.
            #                    error_along_li = step_calculate_output_error_along_li(sli, tci)
            #                    error_along_ti = step_calculate_output_error_along_ti(sli, tci)
            #                    error_along_li.map_with_index { |eli, i| 0.5 * (eli + error_along_ti[i]) }
            #                  when sli < synaptic_layer_index_last && tci == time_col_index_last
            #                    step_calculate_output_error_along_li(sli, tci)
            #                  when sli < synaptic_layer_index_last && tci < time_col_index_last
            #                    # We have 2 errors to deal with; we will average them.
            #                    error_along_li = step_calculate_output_error_along_li(sli, tci)
            #                    error_along_ti = step_calculate_output_error_along_ti(sli, tci)
            #                    error_along_li.map_with_index { |eli, i| 0.5 * (eli + error_along_ti[i]) }
            #                  else
            #                    raise "Index error! (Range Mis-match!) sli: #{sli}, tci: #{tci}"
            #                  end

            #   mns.output_errors = local_errors
            # end

            # private def step_calculate_output_error_along_li(sli, tci)
            #   if sli == synaptic_layer_index_last # && tci < time_col_index_last
            #     mini_net_set[sli][tci].step_calc_output_errors
            #   else
            #     from = 0
            #     to = mini_net_set[sli][tci].width - 1
            #     mini_net_set[sli + 1][tci].input_deltas[from..to]
            #   end
            # end

            # private def step_calculate_output_error_along_ti(sli, tci)
            #   raise "Index error" if tci == time_col_index_last

            #   from = node_input_sizes[sli][tci + 1][:previous_synaptic_layer]
            #   to = from + mini_net_set[sli][tci].width - 1

            #   mini_net_set[sli][tci + 1].input_deltas[from..to]
            # end

            # private def calc_hidden_outputs_expected(sli, tci) # TODO: review and revise?
            #   # This is ONLY valid AFTER 'step_calculate_output_errors_at' is called!!!!
            #   raise "Index Error" if sli == synaptic_layer_index_last

            #   og = mini_net_set[sli][tci].outputs_guessed.clone
            #   oe = mini_net_set[sli][tci].output_errors.clone
            #   og.map_with_index { |o, i| o + oe[i] }
            # end

            # private def step_calculate_output_deltas(sli, tci)
            #   # NOTE: We must use a modified logic compared to MiniNet, which uses:
            #   # ```
            #   # @output_deltas.map_with_index! do |_, i|
            #   #   error = @outputs_expected[i] - @outputs_guessed[i]
            #   #   derivative_propagation_function.call(@outputs_guessed[i]) * error
            #   # end
            #   # ```

            #   mns = mini_net_set[sli][tci]
            #   mns.output_deltas.map_with_index! do |_, i|
            #     mns.derivative_propagation_function.call(mns.outputs_guessed[i].clone) * mns.output_errors[i].clone
            #   end
            # end

            # def final_li_output_error_distances
            #   sli = synaptic_layer_indexes.last
            #   time_col_indexes.map do |tci|
            #     mini_net_set[sli][tci].error_stats.distance
            #   end
            # end
          end
        end
      end
    end
  end
end
