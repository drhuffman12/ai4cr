require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainAndAdjust
          def train(input_sets_given, output_sets_expected) # , until_min_avg_error = 0.1)
            # raise "You are here"

            eval(input_sets_given)

            @layer_range_reversed.map do |h|
              puts "train .. h: #{h}"
              @time_col_range_reversed.map do |t|
                puts "train .. h: #{h}, t: #{t}"

                net = @mini_net_set[h][t]
                # index_size = @mini_net_set[h][t].width
                # outputs_expected_summed = Array.new(index_size) { |i| 0.0 }

                puts "1"

                # add prev layer inputs
                if h == @layer_index_max
                  net.step_load_outputs(output_sets_expected[t])
                  net.step_calculate_output_deltas
                else
                  # input_deltas_from =
                  #   input_deltas_to =
                  net.step_load_output_deltas_from_chained_input_deltas(@mini_net_set[h + 1][t].input_deltas)
                end

                # add prev time col inputs
                if t < @time_col_index_max
                  # input_delta_size = @mini_net_set[h][t+1].outputs_guessed.size
                  input_delta_size = net.outputs_guessed.size
                  input_delta_offset = (h = 0) ? 0 : @mini_net_set[h - 1][t + 1].outputs_guessed.size
                  # input_deltas_from = input_delta_offset
                  input_deltas_to = input_delta_offset + input_delta_size - 1
                  net.append_output_deltas_from_chained_input_deltas(@mini_net_set[h][t + 1].input_deltas[input_delta_offset..input_deltas_to])
                end

                # # A) collect inputs:
                # # outputs_all = Array(Array(Float64)).new
                # # add next layer outputs
                # if h == layer_index_max
                #   outputs_expected_summed.map_with_index! { |d, i| d + output_sets_expected[t][i] }
                #   # outputs_all << output_sets_expected[t]
                # else
                #   index_offset = t > 0 ? @mini_net_set[h + 1][t - 1].outputs_guessed.size : 0
                #   index_to = index_offset + index_size - 1
                #   input_deltas = @mini_net_set[h + 1][t].input_deltas[index_offset..index_to]
                #   outputs_expected_summed.map_with_index! { |d, i| d + input_deltas[i] }
                #   # outputs_all << @mini_net_set[h + 1][t].input_deltas[index_offset..index_to]
                # end

                # puts "2"

                # if t < @time_col_index_max
                #   index_offset = 0
                #   index_to = index_offset + index_size - 1
                #   input_deltas = @mini_net_set[h][t + 1].input_deltas[index_offset..index_to]
                #   outputs_expected_summed.map_with_index! { |d, i| d + input_deltas[i] }
                # end

                # # if h < layer_index_max && t < @time_col_index_max
                # #   # prev_layer_next_time_col_output

                # #   index_from = index_size
                # #   index_to = index_from + index_size - 1

                # #   index_from = (h > 0) ? @mini_net_set[h - 1][t + 1].width : 0
                # #   index_to = index_from + @mini_net_set[h][t].width - 1
                # #   # input_delta_range_max_next_layer = @mini_net_set[h][t].width - 1
                # #   # input_delta_range_max_next_time_col = (@mini_net_set[h + 1][t].width - 1) + input_delta_range_max_next_layer

                # #   outputs_all << @mini_net_set[h][t + 1].input_deltas[index_from..index_to]
                # # end

                # # B) do calc's:
                # puts "3"

                # puts "\n step_load_outputs .. start .. h: #{h}, t: #{t} outputs_expected_summed: #{outputs_expected_summed} \n"

                # net.step_load_inputs(inputs_all.flatten)
                # net.step_load_outputs(outputs_expected_summed)
                # net.step_calculate_output_deltas

                puts "\n step_load_outputs .. done .. h: #{h}, t: #{t} net.outputs_expected: #{net.outputs_expected} \n"

                puts "4"

                net.step_calculate_error

                puts "5"

                net.step_backpropagate

                puts "6"

                error_total
              end
            end
          end

          # # TODO: utilize until_min_avg_error

          # def train(inputs_given, outputs_expected) #, until_min_avg_error = 0.1)
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

          def input_deltas
            h = 0
            @input_deltas = @time_col_range.map do |t|
              @mini_net_set[h][t].input_deltas
            end
          end

          def error_total
            h = @layer_range.last
            @error_total = @time_col_range.map do |t|
              @mini_net_set[h][t].error_total
            end
          end
        end
      end
    end
  end
end
