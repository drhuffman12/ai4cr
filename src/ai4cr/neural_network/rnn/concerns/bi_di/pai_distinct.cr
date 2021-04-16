module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module PaiDistinct
            def calc_node_input_sizes
              if @valid
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |sli|
                  in_size = input_sizes[sli]
                  output_size = node_output_sizes[sli]
                  time_col_indexes.map do |ti|
                    previous_time_column = ti == 0 ? 0 : output_size
                    next_time_column = ti == time_col_indexes.last ? 0 : output_size
                    {
                      channel_forward: {
                        previous_synaptic_layer: in_size,
                        previous_time_column:    previous_time_column,
                        current_self_mem:        output_size,
                      },
                      channel_backward: {
                        previous_synaptic_layer: in_size,
                        next_time_column:        next_time_column,
                        current_self_mem:        output_size,
                      },
                      channel_sl_or_combo: {
                        current_forward:  output_size,
                        current_backward: output_size,
                        current_self_mem: output_size,
                      },
                    }
                  end
                end
              else
                # for type consistency when otherwise invalid:
                [
                  [
                    {
                      channel_forward: {
                        previous_synaptic_layer: 0,
                        previous_time_column:    0,
                        current_self_mem:        0,
                      },
                      channel_backward: {
                        previous_synaptic_layer: 0,
                        next_time_column:        0,
                        current_self_mem:        0,
                      },
                      channel_sl_or_combo: {
                        current_forward:  0,
                        current_backward: 0,
                        current_self_mem: 0,
                      },
                    },
                  ],
                ]
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
