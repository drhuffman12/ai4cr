require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        class Relu < Common::AbstractNet
          def propagation_function
            ->(x : Float64) { x < 0 ? 0.0 : x }
          end

          def derivative_propagation_function
            ->(y : Float64) { y < 0 ? 0.0 : 1.0 }
          end

          def guesses_best
            guesses_ceiled
          end
        end
      end
    end
  end
end
