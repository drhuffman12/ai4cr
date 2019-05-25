
module Ai4cr
  module NeuralNetwork
    class BackpropagationConfig
      include JSON::Serializable

      property structure : Array(Int32)
      property disable_bias : Bool
      property learning_rate : Float64
      property momentum : Float64
    end
  end
end