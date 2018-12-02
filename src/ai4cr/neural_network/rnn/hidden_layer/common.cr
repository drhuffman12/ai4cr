require "./interface"
require "./../node_set/*"
require "./../channel/*"
require "./../weight_set/*"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module HiddenLayer
        class Common(CINS, SINS, PLOC) # , ONS) # Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Common
          include HiddenLayer::Interface

          getter layer_index : Int32
          getter is_first : Bool
          getter bias : Bool # When true, adds hard-coded '1' input and row of weights
          getter time_column_qty : Int32
          getter time_column_range : Range(Int32, Int32)
          getter dendrite_offsets : Array(Int32)
          getter state_qty : Int32
          # getter output_winner_qty : Int32
  
          property previous_layer_output_channel : PLOC
          property channel_local : Channel::Local
          property channel_past : Channel::Past
          property channel_future : Channel::Future
          property channel_combo : Channel::Combo # this layer's output channel

          property weights_local : Array(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, PLOC)) # , SINS, PLOC, ONS))
          # property channel_past : Ai4cr::NeuralNetwork::Rnn::WeightSet::Interface
          # property channel_future : Ai4cr::NeuralNetwork::Rnn::WeightSet::Interface
          # property channel_combo : Ai4cr::NeuralNetwork::Rnn::WeightSet::Interface
  
          def initialize(@previous_layer_output_channel : PLOC, @bias, 
            # @output_winner_qty, 
            @layer_index, @time_column_qty = TIME_COLUMN_QTY_DEFAULT, @dendrite_offsets = Channel::Interface::DENDRITE_OFFSETS_DEFAULT, @state_qty = NodeSet::Interface::STATE_QTY_DEFAULT)
            @is_first = layer_index == 0
  
            @time_column_range = (0..time_column_qty-1)
            
            @channel_local = Channel::Local.new(time_column_qty, dendrite_offsets, state_qty)
            @channel_past = Channel::Past.new(time_column_qty, dendrite_offsets, state_qty)
            @channel_future = Channel::Future.new(time_column_qty, dendrite_offsets, state_qty)
            @channel_combo = Channel::Combo.new(time_column_qty, dendrite_offsets, state_qty)

            # ons = Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden
            @weights_local = time_column_range.map do |time_column_index|
              # Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, SINS, PLOC, ONS).new
              # Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, CINS, PLOC, ons).new
              Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(CINS, PLOC).new(previous_layer_output_channel, channel_local, time_column_index, dendrite_offsets, bias) #, output_winner_qty)
            end
          end
  
          def train
            guess
            correct
          end
  
          def guess
          end
  
          def correct
          end
        end
      end
    end
  end
end
