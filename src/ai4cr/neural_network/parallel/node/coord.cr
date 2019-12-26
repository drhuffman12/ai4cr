require "json"
require "../channel_type"

module Ai4cr
  module NeuralNetwork
    module Parallel
      module Node
        struct Coord
          property channel : ChannelType
          property time_column : Int32
          property layer_index : Int32 # -1 for ChannelType 'Input'

          def is_input?
            channel.is_input?
          end
          def is_hidden?
            channel.is_hidden?
          end
          def is_output?
            channel.is_output?
          end

          # property value
        
          def initialize(@channel = ChannelType::Input, @time_column = 0, @layer_index = -1)
          end
        end
      end
    end
  end
end
