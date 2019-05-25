require "json"

module Ai4cr
  module NeuralNetwork
    module Backpropagation
      struct Config
        include JSON::Serializable

        property structure : Array(Int32)
        property disable_bias : Bool
        property learning_rate : Float64
        property momentum : Float64

        getter input_size # aka height
        getter hidden_layer_sizes # hidden_qty
        getter output_size # aka width

        def initialize(structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
          raise "Invalid Structure" if structure.size == 0
          @structure = structure
          @disable_bias = !!disable_bias
          @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
          @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1
        end

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