require "./node/*"
require "./channel/*"

module Ai4cr
  module NeuralNetwork
    module Rnn # RNN, Bidirectional, Inversable Memory
      class HiddenLayer # Ai4cr::NeuralNetwork::Rnn::HiddenLayer
        getter layer_index : Int32
        getter is_first : Bool
        getter time_column_qty : Int32
        getter time_column_range : Range(Int32, Int32)
        getter dendrite_offsets : Array(Int32)
        getter state_qty : Int32

        property channel_local : Channel::Local
        property channel_past : Channel::Past
        property channel_future : Channel::Future
        property channel_combo : Channel::Combo

        def initialize(@layer_index, @time_column_qty = TIME_COLUMN_QTY_DEFAULT, @dendrite_offsets = cccc, @state_qty = Node::Interface::STATE_QTY_DEFAULT)
          @is_first = layer_index == 0

          @time_column_range = (0..time_column_qty-1)
          
          @channel_local = Channel::Local.new(time_column_qty, dendrite_offsets, state_qty)
          @channel_past = Channel::Past.new(time_column_qty, dendrite_offsets, state_qty)
          @channel_future = Channel::Future.new(time_column_qty, dendrite_offsets, state_qty)
          @channel_combo = Channel::Combo.new(time_column_qty, dendrite_offsets, state_qty)
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
