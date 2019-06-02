# Ai4cr::NeuralNetwork::Backpropagation::Common::Initializers::Momentum

module Ai4cr
  module NeuralNetwork
    module Backpropagation
      module Common
        module Initializers
          module Momentum
            DEFAULT_MOMENTUM = 0.1
            
            property momentum : Float64
  
            def init_momentum(_momentum)
              # must be positive
              _momentum && _momentum.as(Float64) > 0.0 ? _momentum.as(Float64) : DEFAULT_MOMENTUM
            end
          end
        end
      end
    end
  end
end
