
require "json"

module Ai4cr
  module NeuralNetwork
    module Parallel
      enum Channel
        Input  = -1
        Past   =  0
        Local  =  1
        Future =  2
        Combo  =  3 # aka Output

        def is_input?
          self == ChannelType::Input
        end

        def is_hidden?
          self == !is_input? && !is_output?
        end

        def is_output?
          self == ChannelType::Combo
        end
      end
    end
  end
end


