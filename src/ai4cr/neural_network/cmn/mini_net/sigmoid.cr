require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        class Sigmoid < Common::AbstractNet
          def propagation_function
            ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
          end

          def derivative_propagation_function
            ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
          end

          def guesses_best
            guesses_rounded
          end
        end
      end
    end
  end
end
