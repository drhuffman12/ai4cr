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
        module SidePast(CINS)
          include WeightSet::Interface

          property side_past_input_node_sets : Array(CINS)
          property time_column_index_past_offsets : Array(Int32)
          property weights_side_past : Array(WeightsSimple)

          def init_side_past(time_column_index, dendrite_offsets, previous_layer_output_channel, bias, output_node_set)
            @time_column_index_past_offsets = (dendrite_offsets.reverse.map{|d| -d}).reject{|i| time_column_index + i < 0 || time_column_index + i > previous_layer_output_channel.time_column_range.max}

            @side_past_input_node_sets = time_column_index_past_offsets.map do |offset|
              previous_layer_output_channel.node_sets[offset]
            end


            side_past_input_node_sets.each do |node_set|
              @weights_side_past << output_node_set.state_values.map do |outs|
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
