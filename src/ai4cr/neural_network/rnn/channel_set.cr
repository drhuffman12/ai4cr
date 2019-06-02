require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct ChannelSet
        include JSON::Serializable
        # include NeuralNetwork::Common::Initializers::LearningRate
        # include NeuralNetwork::Common::Initializers::Momentum

        # property qty_states_in
        # property qty_states_out
        # property qty_time_cols
        # property qty_lpfc_layers
        # property qty_recent_memory

        # property structure_hidden_laters : Array(Int32)
        property channel_set_index : Int32
        property disable_bias : Bool
        property config : Config

        # property local  : Array(NeuralNetwork::Backpropagation::Net)
        # property past   : Array(NeuralNetwork::Backpropagation::Net)
        # property future : Array(NeuralNetwork::Backpropagation::Net)
        # property combo  : Array(NeuralNetwork::Backpropagation::Net)

        

        def initialize(@channel_set_index, @disable_bias, @config)
          # @local = init_local
          # @past = init_past
          # @future = init_future
          # @combo = init_combo
        end

        # def init_local
        #   (0..(qty_time_cols - 1)).to_a.map do |time_col|

        #   end
        # end

        # def init_past
        # end

        # def init_future
        # end

        # def init_combo
        # end
      end
    end
  end
end
