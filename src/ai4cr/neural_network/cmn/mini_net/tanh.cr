require "json"
require "./common.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        class Tanh
          include JSON::Serializable
          include Common

          def propagation_function
            ->(x : Float64) { Math.tanh(x) }
          end

          def derivative_propagation_function
            ->(y : Float64) { 1.0 - (y**2) }
          end

          def guesses_best
            guesses_rounded
          end
        end
      end
    end
  end
end

# puts Ai4cr::NeuralNetwork::Cmn::MiniNet::.new(2,3).to_json
