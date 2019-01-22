require "./../node_set/*"
require "./interface"
require "./center"
require "./side_past"
require "./side_future"
require "./memory"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        # CINS for center_input_node_set
        # PLOC for previous_layer_output_channel
        class Past(CINS, PLOC)
          include WeightSet::Interface
          include WeightSet::Center(CINS)
          include WeightSet::SidePast(CINS)
          # include WeightSet::SideFuture(CINS)
          include WeightSet::Memory
          
          property previous_layer_output_channel : PLOC
          property output_channel : Ai4cr::NeuralNetwork::Rnn::Channel::Local
          property time_column_index : Int32
          property dendrite_offsets : Array(Int32)
          property bias : Bool
          # property output_winner_qty : Int32

          property output_node_set : Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden

          def initialize(
            # @hidden_layer_index,
            @previous_layer_output_channel,
            @output_channel,
            @time_column_index,
            @dendrite_offsets = Channel::Interface::DENDRITE_OFFSETS_DEFAULT,
            @bias = false,
            # @output_winner_qty = 1
          )            
            # keep #initialize happy:
            @center_input_node_set = CINS.new
            @time_column_index_past_offsets = Array(Int32).new
            # @time_column_index_future_offsets = Array(Int32).new
            @side_past_input_node_sets = Array(CINS).new
            # @side_future_input_node_sets = Array(CINS).new

            @weights_center = WeightsSimple.new
            @weights_side_past = Array(WeightsSimple).new
            # @weights_side_future = Array(WeightsSimple).new
            @weights_memory = Array(WeightsSimple).new

            # actual initializers #initialize happy:
            @output_node_set = output_channel.node_sets[time_column_index]

            init_center(time_column_index, previous_layer_output_channel, bias, output_node_set)
            init_side_past(time_column_index, dendrite_offsets, previous_layer_output_channel, bias, output_node_set)
            # init_side_future(time_column_index, dendrite_offsets, previous_layer_output_channel, bias, output_node_set)
            init_memory(time_column_index, output_node_set)
          end
        end
      end
    end
  end
end
