module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module PaiDistinct
            # 'Pai' aka short for 'PropsAndInits'

            alias NodeInputSizes = Array(Array(NamedTuple(
              channel_forward: NamedTuple(
                # enabled: Bool,
                current_self_mem: Int32,
                previous_synaptic_layer_channel_sl_or_combo: Int32,
                previous_synaptic_layer_channel_forward: Int32,
                previous_time_column: Int32),
              channel_backward: NamedTuple(
                # enabled: Bool,
                current_self_mem: Int32,
                previous_synaptic_layer_channel_sl_or_combo: Int32,
                previous_synaptic_layer_channel_backward: Int32,
                next_time_column: Int32),
              channel_sl_or_combo: NamedTuple(
                current_self_mem: Int32,
                previous_synaptic_layer_channel_sl_or_combo: Int32,
                current_forward: Int32, current_backward: Int32))))
            # alias MiniNetSet = Array(Array(Hash(Symbol, Hash(Symbol, Int32))))
            # alias MiniNetSet = Array(Array(Ai4cr::NeuralNetwork::Cmn::MiniNet)
            alias MiniNetSet = Array(Array(Hash(Symbol, Cmn::MiniNet)))

            property node_input_sizes = NodeInputSizes.new
            property mini_net_set = MiniNetSet.new

            # ameba:disable Metrics/CyclomaticComplexity
            def calc_node_input_sizes
              if @valid
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |sli|
                  in_size = input_sizes[sli]
                  output_size = node_output_sizes[sli]
                  time_col_indexes.map do |ti|
                    previous_time_column = ti == 0 ? 0 : output_size

                    # previous_channel_forward_or_backward = ti == 0 ? 0 : node_output_sizes[sli - 1] # in_size : output_size
                    previous_channel_forward_or_backward = [0, 1].includes?(sli) ? 0 : hidden_size # node_output_sizes[sli - 1]

                    # previous_synaptic_layer_channel_sl_or_combo = ti == 0 ? 0 : node_output_sizes[sli - 1]     # in_size # ti == 0 ? in_size : output_size

                    previous_synaptic_layer_channel_sl_or_combo = sli == 0 ? in_size : hidden_size
                    # previous_synaptic_layer_channel_sl_or_combo = case
                    #   when sli == 0
                    #     in_size
                    #   # when sli == 1
                    #   #   node_output_sizes[sli - 1] # 0
                    #   else
                    #     hidden_size # node_output_sizes[sli - 1]
                    #   end

                    next_time_column = ti == time_col_indexes.last ? 0 : output_size
                    {
                      channel_forward: {
                        # enabled: sli > 0,
                        current_self_mem:                            sli == 0 ? 0 : output_size,
                        previous_synaptic_layer_channel_sl_or_combo: sli == 0 ? 0 : previous_synaptic_layer_channel_sl_or_combo, # sli == 0 ? 0 : in_size,
                        previous_synaptic_layer_channel_forward:     previous_channel_forward_or_backward,                       # sli == 0 ? 0 : previous_channel_forward_or_backward,
                        previous_time_column:                        sli == 0 ? 0 : previous_time_column,
                      },
                      channel_backward: {
                        # enabled: sli > 0,
                        current_self_mem:                            sli == 0 ? 0 : output_size,
                        previous_synaptic_layer_channel_sl_or_combo: sli == 0 ? 0 : previous_synaptic_layer_channel_sl_or_combo, # sli == 0 ? 0 : in_size,
                        previous_synaptic_layer_channel_backward:    previous_channel_forward_or_backward,                       # sli == 0 ? 0 : previous_channel_forward_or_backward,
                        next_time_column:                            sli == 0 ? 0 : next_time_column,
                      },
                      channel_sl_or_combo: {
                        # disabled: false,
                        current_self_mem:                            output_size,
                        previous_synaptic_layer_channel_sl_or_combo: previous_synaptic_layer_channel_sl_or_combo, # sli == 0 ? in_size : node_output_sizes[sli - 1], # previous_synaptic_layer_channel_sl_or_combo,
                        current_forward:                             sli == 0 ? 0 : output_size,
                        current_backward:                            sli == 0 ? 0 : output_size,
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
              # TODO
              # mns = MiniNetSet.new
              synaptic_layer_indexes.map do |sli|
                # NOTE: It should suffice to have bias only on the first sli nets.
                #   So, force bias only on 1st and none on others
                sli_gt_0 = sli != 0

                mn_output_size = node_output_sizes[sli]
                time_col_indexes.map do |ti|
                  # mn_input_size = node_input_sizes[sli][ti].values.sum

                  # Alternate thru the sequence of learning styles
                  lsi = sli % @learning_styles.size

                  h = Hash(Symbol, Ai4cr::NeuralNetwork::Cmn::MiniNet).new

                  if sli_gt_0
                    mn_input_size_forward = node_input_sizes[sli][ti][:channel_forward].values.sum
                    h[:channel_forward] = Cmn::MiniNet.new(
                      height: mn_input_size_forward,
                      width: mn_output_size,

                      learning_styles: @learning_styles[lsi],
                      deriv_scale: @deriv_scale,

                      bias_disabled: sli_gt_0,
                      bias_default: @bias_default,

                      learning_rate: @learning_rate,
                      momentum: @momentum,

                      weight_init_scale: @weight_init_scale
                    )

                    mn_input_size_backward = node_input_sizes[sli][ti][:channel_backward].values.sum
                    h[:channel_backward] = Cmn::MiniNet.new(
                      height: mn_input_size_backward,
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

                  mn_input_size_sl_or_combo = node_input_sizes[sli][ti][:channel_sl_or_combo].values.sum
                  h[:channel_forward] = Cmn::MiniNet.new(
                    height: mn_input_size_sl_or_combo,
                    width: mn_output_size,

                    learning_styles: @learning_styles[lsi],
                    deriv_scale: @deriv_scale,

                    bias_disabled: sli_gt_0,
                    bias_default: @bias_default,

                    learning_rate: @learning_rate,
                    momentum: @momentum,

                    weight_init_scale: @weight_init_scale
                  )

                  h
                end
              end
            end
          end
        end
      end
    end
  end
end
