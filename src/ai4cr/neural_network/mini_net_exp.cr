require "json"
require "./mini_net_common.cr"
# require "./src/ai4cr/neural_network/mini_net_common.cr"
# require "mini_net_common.cr"


module Ai4cr
  module NeuralNetwork
    struct MiniNetExp < MiniNetCommon

      ####
      # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
      def propagation_function
        ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
      end

      # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
      def derivative_propagation_function
        ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
      end
      ####
      
    end
  end
end

# puts Ai4cr::NeuralNetwork::MiniNet.new(2,3).to_json
