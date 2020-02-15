require "json"
require "./common.cr"
# require "./src/ai4cr/neural_network/mini_net_common.cr"
# require "mini_net_common.cr"

require "json"
require "./common.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        struct Tanh < Common  
          def propagation_function
            ->(x : Float64) { Math.tanh(x)}
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
