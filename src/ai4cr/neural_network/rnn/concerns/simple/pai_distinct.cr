module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module Simple
          module PaiDistinct
            alias NodeInputSizes = Array(Array(NamedTuple(
              previous_synaptic_layer: Int32,
              previous_time_column: Int32)))

            property node_input_sizes = NodeInputSizes.new

            def calc_node_input_sizes
              if @valid
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |li|
                  in_size = input_sizes[li]
                  output_size = node_output_sizes[li]
                  time_col_indexes.map do |ti|
                    if ti == 0
                      {previous_synaptic_layer: in_size, previous_time_column: 0}
                    else
                      {previous_synaptic_layer: in_size, previous_time_column: output_size}
                    end
                  end
                end
              else
                [[{previous_synaptic_layer: 0, previous_time_column: 0}]]
              end
            end

            def init_mini_net_set
              synaptic_layer_indexes.map do |li|
                # NOTE: It should suffice to have bias only on the first li nets.
                #   So, force bias only on 1st and none on others
                li_gt_0 = li != 0

                mn_output_size = node_output_sizes[li]
                time_col_indexes.map do |ti|
                  mn_input_size = node_input_sizes[li][ti].values.sum

                  # Alternate thru the sequence of learning styles
                  lsi = li % @learning_styles.size

                  Cmn::MiniNet.new(
                    height: mn_input_size,
                    width: mn_output_size,

                    learning_styles: @learning_styles[lsi],
                    deriv_scale: @deriv_scale,

                    bias_disabled: li_gt_0,
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
  end
end
