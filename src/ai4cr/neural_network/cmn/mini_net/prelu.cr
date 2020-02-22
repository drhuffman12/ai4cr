require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        class Prelu < Common::AbstractNet
          @deriv_scale = 0.001

          def set_deriv_scale(scale)
            @deriv_scale = scale
          end

          def propagation_function
            ->(x : Float64) { x < 0 ? 0.0 : x }
          end

          def derivative_propagation_function
            ->(y : Float64) { y < 0 ? @deriv_scale : 1.0 }
          end

          def guesses_best
            guesses_ceiled
          end
        end
      end
    end
  end
end
