require "./../node_set/*"
# require "./../channel/*"
# require "./../hidden_layer/*"
require "./interface"
require "./../math"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        abstract class Common(CINS, SINS, ONS) # Ai4cr::NeuralNetwork::Rnn::WeightSet::Common
          include WeightSet::Interface

          alias IOSimple = Array(Float64)
          alias WeightsSimple = Array(IOSimple)

          # property hidden_layer_index : Int32
          # property input_node_sets : Array(IOSimple)
          property center_input_node_set : CINS
          property side_input_node_sets : Array(SINS) # Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Interface)
          property output_node_set : ONS # Ai4cr::NeuralNetwork::Rnn::NodeSet::Interface
          property weights : Array(WeightsSimple)
          # property weights_center : WeightsSimple
          # property weights_sides : Array(WeightsSimple)
          property bias : Bool
          property output_winner_qty : Int32
          
          def initialize(
            # @hidden_layer_index,
            @center_input_node_set,
            @side_input_node_sets,
            @output_node_set,
            # @hidden_layer_index, @time_column_index, @memory_layer_qty = 1,
            # @output_state_qty = 3, @hidden_state_qty = 4, @input_state_qty = 2,
            # @dendrite_offsets = Channel::Interface::DENDRITE_OFFSETS_DEFAULT,
            @bias = false,
            @output_winner_qty = 1 # when guessing, exaggerate top n number of output states to maximum; others get minimized
          )

            @weights = Array(WeightsSimple).new
            # @weights_center = WeightsSimple.new
            # @weights_sides = Array(WeightsSimple).new
            init_weights
          end

          def init_weights
            # @weights = output_node_set.map do |outs|
            #   side_input_node_sets.map do |inputs|
            #     w = inputs.map do |ins|
            #       Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
            #     end
            #     w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
            #   end
            # end

            # @input_node_sets = Array(IOSimple).new
            @weights << output_node_set.state_values.map do |outs|
              w = center_input_node_set.state_values.map do |inputs|
                Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
              end
              w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
              w
            end
            # side_input_node_sets.map do |node_set|
            #   w = node_set.state_values.map do |inputs|
            #     Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
            #   end
            #   w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
            #   @weights << w
            # end

            side_input_node_sets.each do |node_set|
              @weights << output_node_set.state_values.map do |outs|
                w = node_set.state_values.map do |inputs|
                  Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
                end
                w << Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one if bias
                w
              end
            end

            # if hidden_layer_index > 0
          end
        end
      end
    end
  end
end
