require "./interface"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        module Center (CINS)
          include WeightSet::Interface
          
          property center_input_node_set : CINS
          property weights_center_input_node_set : WeightsSimple

          def init_center(time_column_index, previous_layer_output_channel, bias, output_node_set)
            @center_input_node_set = previous_layer_output_channel.node_sets[time_column_index]

            @weights_center_input_node_set = output_node_set.state_values.map do |outs|
              w = center_input_node_set.state_values.map do |inputs|
                Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
              end
              w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
              w
            end
          end

          def forward_sums_center
            weights_center_input_node_set.map_with_index do |outs, output_index|
              s = center_input_node_set.state_values.map_with_index do |inputs, input_index|
                center_input_node_set.state_values[input_index] * weights_center_input_node_set[output_index][input_index]
              end
              s << weights_center_input_node_set[output_index][center_input_node_set.state_qty] if bias
              s
            end
          end
        end
      end
    end
  end
end
