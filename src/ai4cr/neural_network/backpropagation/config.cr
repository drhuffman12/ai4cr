require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Backpropagation
      struct Config
        include JSON::Serializable
        include NeuralNetwork::Common::Initializers::LearningRate
        include NeuralNetwork::Common::Initializers::Momentum

        property structure : Array(Int32)
        property disable_bias : Bool
        # property learning_rate : Float64
        # property momentum : Float64

        getter input_size # aka height
        getter hidden_layer_sizes # hidden_qty
        getter output_size # aka width

        # def initialize(structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
        def initialize(structure, disable_bias = true, learning_rate = nil, momentum = nil)
          raise "Invalid Structure" if structure.size == 0
          @structure = structure
          @disable_bias = !!disable_bias
          @learning_rate = init_learning_rate(learning_rate)
          @momentum = init_momentum(momentum)
        end

        # def init_learning_rate(_learning_rate)
        #   # must be positive
        #   _learning_rate.nil? || _learning_rate.as(Float64) <= 0.0 ? DEFAULT_LEARNING_RATE : _learning_rate.as(Float64)
        # end

        # def init_momentum(_momentum)
        #   # must be positive
        #   _momentum && _momentum.as(Float64) > 0.0 ? _momentum.as(Float64) : DEFAULT_MOMENTUM
        # end

        def input_size # aka height
          structure.first
        end

        # def hidden_qty
        def hidden_layer_sizes
          structure[1..-2]
        end

        def output_size # aka width
          structure.last
        end
      end
    end
  end
end