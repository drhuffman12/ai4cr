require "./hidden"
require "./../node_set/*"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        class Local(CINS, SINS, PLOC) < Hidden(CINS, CINS, NodeSet::Hidden) # SINS) # Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, SINS, PLOC)
          # TIME_COLUMN_QTY_DEFAULT = 3
          # DENDRITE_OFFSETS_DEFAULT = [1] # [1,2,3,4,5,6,7,8] # [1,2,4,8,16,32,64,128]
          # property previous_layer_output_channel : PLOC
          property time_column_index_offsets : Array(Int32)
          property time_column_index : Int32

          def initialize(
            # @hidden_layer_index,
            @previous_layer_output_channel : PLOC, # Ai4cr::NeuralNetwork::Rnn::Channel::Input or Ai4cr::NeuralNetwork::Rnn::Channel::Combo
            @output_channel : Ai4cr::NeuralNetwork::Rnn::Channel::Local,
            @time_column_index,
            @dendrite_offsets = Channel::Interface::DENDRITE_OFFSETS_DEFAULT,
            # @side_input_node_sets,
            # @outputs,
            @bias = false,
            @output_winner_qty = 1
          )
            @center_input_node_set = previous_layer_output_channel.node_sets[time_column_index]
            offset_max = previous_layer_output_channel.time_column_range.max
            # offsets_tmp = dendrite_offsets.reverse.map{|d| -d} + [0] + dendrite_offsets
            offsets_tmp = dendrite_offsets.reverse.map{|d| -d} + dendrite_offsets
            @time_column_index_offsets = (offsets_tmp).reject{|i| time_column_index + i < 0 || time_column_index + i > offset_max}
            @side_input_node_sets = time_column_index_offsets.map do |offset|
              previous_layer_output_channel.node_sets[offset]
            end
            @output_node_set = output_channel.node_sets[time_column_index]
            # super(hidden_layer_index, center_input_node_set, side_input_node_sets, output_node_set, bias, output_winner_qty)
            super(center_input_node_set, side_input_node_sets, output_node_set, bias, output_winner_qty)
          end
        end
      end
    end
  end
end
