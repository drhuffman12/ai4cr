require "./../node_set/*"
# require "./../channel/*"
# require "./../hidden_layer/*"
require "./interface"
require "./../math"

# require "./hidden"
# require "./../node_set/*"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        module SideFuture(CINS)
          include WeightSet::Interface

          property side_future_input_node_sets : Array(CINS)
          property time_column_index_future_offsets : Array(Int32)
          property weights_side_future : Array(WeightsSimple)

          def init_side_future(time_column_index, dendrite_offsets, previous_layer_output_channel, output_node_set)
            @time_column_index_future_offsets = dendrite_offsets.reject{|i| time_column_index + i < 0 || time_column_index + i > previous_layer_output_channel.time_column_range.max}

            @side_future_input_node_sets = time_column_index_future_offsets.map do |offset|
              previous_layer_output_channel.node_sets[offset]
            end

            side_future_input_node_sets.each do |node_set|
              @weights_side_future << output_node_set.state_values.map do |outs|
                w = node_set.state_values.map do |inputs|
                  Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
                end
                w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
                w
              end
            end
          end
        end
      end
    end
  end
end
