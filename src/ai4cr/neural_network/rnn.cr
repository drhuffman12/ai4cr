# require "./rnn/*"

module Ai4cr
  module NeuralNetwork
    module Rnn


      enum ChannelType
        Local  = 0
        Past   = 1
        Future = 2
        Combo  = 3
        Input  = 4
        Output = 5
        Memory = 6
      end

      # alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: ChannelType, time_col_index: Int32)
      alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: Int32, time_col_index: Int32)

    end
  end
end

require "./rnn/*"
