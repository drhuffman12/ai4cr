# require "./rnn/*"

module Ai4cr
  module NeuralNetwork
    module Rnn


      enum ChannelType
        Local
        Past
        Future
        Combo
        Input
        Output
      end

      # alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: ChannelType, time_col_index: Int32)
      alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: Int32, time_col_index: Int32)

    end
  end
end

require "./rnn/*"
