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
        class Local(CINS, PLOC) # (CINS, SINS, PLOC, ONS) # < Hidden(CINS, CINS, NodeSet::Hidden) # SINS) # Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, SINS, PLOC)
          include WeightSet::Interface

          alias IOSimple = Array(Float64)
          alias WeightsSimple = Array(IOSimple)

          # TIME_COLUMN_QTY_DEFAULT = 3
          # DENDRITE_OFFSETS_DEFAULT = [1] # [1,2,3,4,5,6,7,8] # [1,2,4,8,16,32,64,128]
          # property previous_layer_output_channel : PLOC
          # property time_column_index_offsets : Array(Int32)
          property time_column_index_past_offsets : Array(Int32)
          property time_column_index_future_offsets : Array(Int32)
          property time_column_index : Int32

          property center_input_node_set : CINS
          # property side_input_node_sets : Array(CINS) # Array
          property side_past_input_node_sets : Array(CINS) # Array
          property side_future_input_node_sets : Array(CINS) # Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Interface)
          property output_node_set : Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden # Ai4cr::NeuralNetwork::Rnn::NodeSet::Interface

          property bias : Bool
          property output_winner_qty : Int32

          # memory
          property weights_center_input_node_set : Array(WeightsSimple)
          # property weights_sides : Array(WeightsSimple)
          property weights_side_past : Array(WeightsSimple)
          property weights_side_future : Array(WeightsSimple)
          property weights_memory : Array(WeightsSimple)

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
            # offsets_tmp = dendrite_offsets.reverse.map{|d| -d} + dendrite_offsets
            # @time_column_index_offsets = (offsets_tmp).reject{|i| time_column_index + i < 0 || time_column_index + i > offset_max}
            @time_column_index_past_offsets = (dendrite_offsets.reverse.map{|d| -d}).reject{|i| time_column_index + i < 0 || time_column_index + i > offset_max}
            #   previous_layer_output_channel.node_sets[offset]
            # end
            @time_column_index_future_offsets = (dendrite_offsets).reject{|i| time_column_index + i < 0 || time_column_index + i > offset_max}
            #   previous_layer_output_channel.node_sets[offset]
            # end
            @side_past_input_node_sets = time_column_index_past_offsets.map do |offset|
              previous_layer_output_channel.node_sets[offset]
            end
            @side_future_input_node_sets = time_column_index_future_offsets.map do |offset|
              previous_layer_output_channel.node_sets[offset]
            end
            @output_node_set = output_channel.node_sets[time_column_index]
            # super(hidden_layer_index, center_input_node_set, side_input_node_sets, output_node_set, bias, output_winner_qty)
            # super(center_input_node_set, side_input_node_sets, output_node_set, bias, output_winner_qty)

            @weights_center_input_node_set = Array(WeightsSimple).new
            @weights_side_past = Array(WeightsSimple).new
            @weights_side_future = Array(WeightsSimple).new
            @weights_memory = Array(WeightsSimple).new
            # @weights_center = WeightsSimple.new
            # @weights_sides = Array(WeightsSimple).new
            init_weights
          end

          def init_weights
            # super
            @weights_center_input_node_set << output_node_set.state_values.map do |outs|
              w = center_input_node_set.state_values.map do |inputs|
                Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
              end
              w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
              w
            end

            # weights_sides
            side_past_input_node_sets.each do |node_set|
              @weights_side_past << output_node_set.state_values.map do |outs|
                w = node_set.state_values.map do |inputs|
                  Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
                end
                w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
                w
              end
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

            # hidden
            output_node_set.memory_values_set.each do |memory_values|
              @weights_memory << output_node_set.state_values.map do |outs|
                w = memory_values.map do |inputs|
                  Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
                end
                w
              end
            end
          end
        end
      end
    end
  end
end
