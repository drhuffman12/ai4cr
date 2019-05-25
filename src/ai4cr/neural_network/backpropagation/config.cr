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

        def initialize(@structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
          @disable_bias = !!disable_bias
          @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
          @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1
        end
      end
    end
  end
end