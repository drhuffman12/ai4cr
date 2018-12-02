require "./interface"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        module Center (CINS)
          include WeightSet::Interface
          
          property center_input_node_set : CINS
          property weights_center : WeightsSimple

          def init_center(time_column_index, previous_layer_output_channel, bias, output_node_set)
            @center_input_node_set = previous_layer_output_channel.node_sets[time_column_index]

            @weights_center = output_node_set.state_values.map do |outs|
              w = center_input_node_set.state_values.map do |inputs|
                Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
              end
              w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
              w
            end
          end

          def forward_products_center
            weights_center.map_with_index do |outs, output_index|
              products_partial = center_input_node_set.state_values.map_with_index do |inputs, input_index|
                center_input_node_set.state_values[input_index] * weights_center[output_index][input_index]
              end
              products_partial << weights_center[output_index][center_input_node_set.state_qty] if bias
              products_partial
            end
          end
        end
      end
    end
  end
end
