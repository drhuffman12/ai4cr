module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module PaiDistinct
            alias NodeInputSizes = Array(Array(NamedTuple(
              channel_forward: NamedTuple(
                current_self_mem: Int32,
                previous_synaptic_layer_inputs_or_combo: Int32,
                previous_synaptic_layer_channel_forward: Int32,
                previous_time_column: Int32),
              channel_backward: NamedTuple(
                current_self_mem: Int32,
                previous_synaptic_layer_inputs_or_combo: Int32,
                previous_synaptic_layer_channel_backward: Int32,
                next_time_column: Int32),
              channel_sl_or_combo: NamedTuple(
                current_self_mem: Int32,
                previous_synaptic_layer_inputs_or_combo: Int32,
                current_forward: Int32, current_backward: Int32))))

            property node_input_sizes = NodeInputSizes.new

            def calc_node_input_sizes
              if @valid
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |sli|
                  in_size = input_sizes[sli]
                  output_size = node_output_sizes[sli]
                  time_col_indexes.map do |ti|
                    previous_time_column = ti == 0 ? 0 : output_size
                    previous_channel_forward_or_backward = ti == 0 ? in_size : output_size
                    previous_channel_inputs_or_combo = ti == 0 ? 0 : output_size
                    next_time_column = ti == time_col_indexes.last ? 0 : output_size
                    {
                      channel_forward: {
                        current_self_mem:                        output_size,
                        previous_synaptic_layer_inputs_or_combo: in_size,
                        previous_synaptic_layer_channel_forward: previous_channel_forward_or_backward,
                        previous_time_column:                    previous_time_column,
                      },
                      channel_backward: {
                        current_self_mem:                         output_size,
                        previous_synaptic_layer_inputs_or_combo:  in_size,
                        previous_synaptic_layer_channel_backward: previous_channel_forward_or_backward,
                        next_time_column:                         next_time_column,
                      },
                      channel_sl_or_combo: {
                        current_self_mem:                        output_size,
                        previous_synaptic_layer_inputs_or_combo: previous_channel_inputs_or_combo,
                        current_forward:                         output_size,
                        current_backward:                        output_size,
                      },
                    }
                  end
                end
              else
                # for type consistency when otherwise invalid:
                NodeInputSizes.new
              end
            end

            # def init_mini_net_set
            #   # TODO
            #   synaptic_layer_indexes.map do |li|
            #     # NOTE: It should suffice to have bias only on the first li nets.
            #     #   So, force bias only on 1st and none on others
            #     li_gt_0 = li != 0

            #     mn_output_size = node_output_sizes[li]
            #     time_col_indexes.map do |ti|
            #       mn_input_size = node_input_sizes[li][ti].values.sum

            #       # Alternate thru the sequence of learning styles
            #       lsi = li % @learning_styles.size

            #       Cmn::MiniNet.new(
            #         height: mn_input_size,
            #         width: mn_output_size,

            #         learning_styles: @learning_styles[lsi],
            #         deriv_scale: @deriv_scale,

            #         bias_disabled: li_gt_0,
            #         bias_default: @bias_default,

            #         learning_rate: @learning_rate,
            #         momentum: @momentum,

            #         weight_init_scale: @weight_init_scale
            #       )
            #     end
            #   end
            # end

          end
        end
      end
    end
  end
end
