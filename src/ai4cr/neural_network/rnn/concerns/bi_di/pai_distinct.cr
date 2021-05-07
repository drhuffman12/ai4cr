module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module PaiDistinct
            # 'Pai' aka short for 'PropsAndInits'

            property node_input_sizes = NodeInputSizes.new
            property mini_net_set = MiniNetSet.new

            # ameba:disable Metrics/CyclomaticComplexity
            def calc_node_input_sizes
              if @valid
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |sli|
                  in_size = input_sizes[sli]
                  output_size = node_output_sizes[sli]
                  time_col_indexes.map do |tci|
                    {
                      channel_forward: {
                        # enabled: sli > 0,
                        current_self_mem:            sli == 0 ? 0 : hidden_size,
                        sl_previous_input_or_combo:  sli == 0 ? 0 : in_size,
                        sl_previous_channel_forward: sli <= 1 ? 0 : hidden_size,
                        tc_previous_channel_forward: sli == 0 ? 0 : (tci == 0 ? 0 : hidden_size),
                      },
                      channel_backward: {
                        # enabled: sli > 0,
                        current_self_mem:             sli == 0 ? 0 : hidden_size,
                        sl_previous_input_or_combo:   sli == 0 ? 0 : in_size,
                        sl_previous_channel_backward: sli <= 1 ? 0 : hidden_size,
                        tc_next_channel_backward:     sli == 0 ? 0 : (tci == time_col_indexes.last ? 0 : hidden_size),
                      },
                      channel_sl_or_combo: {
                        # disabled: false,
                        current_self_mem:           output_size,
                        sl_previous_input_or_combo: in_size,
                        current_forward:            sli == 0 ? 0 : hidden_size,
                        current_backward:           sli == 0 ? 0 : hidden_size,
                      },
                    }
                  end
                end
              else
                # for type consistency when otherwise invalid:
                NodeInputSizes.new
              end
            end

            # ameba:enable Metrics/CyclomaticComplexity

            def init_mini_net_set
              sli_last = synaptic_layer_indexes.last
              synaptic_layer_indexes.map do |sli|
                # NOTE: It should suffice to have bias only on the first sli nets.
                #   So, force bias only on 1st and none on others
                sli_gt_0 = sli != 0

                # Alternate thru the sequence of learning styles
                lsi = sli % @learning_styles.size

                # mn_output_size = node_output_sizes[sli]
                time_col_indexes.map do |tci|
                  h = Hash(Symbol, Ai4cr::NeuralNetwork::Cmn::MiniNet).new
                  if sli == 0 # && channel_symbol != :channel_sl_or_combo
                    # ONLY add mini_net for channel :channel_sl_or_combo
                    h[:channel_sl_or_combo] = gen_mini_net_4_channel_sl_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
                  else
                    # Add mini_net for all channels
                    h[:channel_forward] = gen_mini_net_4_channel_forward(sli, tci, sli_gt_0, lsi)
                    h[:channel_backward] = gen_mini_net_4_channel_backward(sli, tci, sli_gt_0, lsi)
                    h[:channel_sl_or_combo] = gen_mini_net_4_channel_sl_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
                  end

                  h
                end
              end
            end

            def gen_mini_net_4_channel_forward(sli, tci, sli_gt_0, lsi)
              mn_input_size = node_input_sizes[sli][tci][:channel_forward].values.sum
              mn_output_size = hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net_4_channel_backward(sli, tci, sli_gt_0, lsi)
              mn_input_size = node_input_sizes[sli][tci][:channel_backward].values.sum
              mn_output_size = hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net_4_channel_sl_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
              mn_input_size = node_input_sizes[sli][tci][:channel_sl_or_combo].values.sum
              mn_output_size = (sli == sli_last) ? output_size : hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
              Cmn::MiniNet.new(
                height: mn_input_size,
                width: mn_output_size,

                learning_styles: @learning_styles[lsi],
                deriv_scale: @deriv_scale,

                bias_disabled: sli_gt_0,
                bias_default: @bias_default,

                learning_rate: @learning_rate,
                momentum: @momentum,

                weight_init_scale: @weight_init_scale
              )
            end
          end
        end
      end
    end
  end
end
