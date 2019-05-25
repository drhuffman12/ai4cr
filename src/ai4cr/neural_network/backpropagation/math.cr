module Ai4cr
  module NeuralNetwork
    module Backpropagation
      module Math
        def initial_weight_function
          ->(n : Int32, i : Int32, j : Int32) { ((rand(2000))/1000.0) - 1 }
        end

        def propagation_function
          ->(x : Float64) { 1/(1 + ::Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
        end

        def derivative_propagation_function
          ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
        end
      end
    end
  end
end