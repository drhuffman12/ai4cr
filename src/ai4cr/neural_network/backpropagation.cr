# Ported By:: Daniel Huffman
# Url::       https://github.com/drhuffman12/ai4cr
#
# Based on::  Ai4r
#   Author::  Sergio Fierens
#   License:: MPL 1.1
#   Project:: ai4r
#   Url::     http://ai4r.org/
#   Githun::  https://github.com/SergioFierens/ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt
module Ai4cr
  # Artificial Neural Networks are mathematical or computational models based on
  # biological neural networks.
  #
  # More about neural networks:
  #
  # * http://en.wikipedia.org/wiki/Artificial_neural_network
  #
  module NeuralNetwork
    # = Introduction
    #
    # This is an implementation of a multilayer perceptron network, using
    # the backpropagation algorithm for learning.
    #
    # Backpropagation is a supervised learning technique (described
    # by Paul Werbos in 1974, and further developed by David E.
    # Rumelhart, Geoffrey E. Hinton and Ronald J. Williams in 1986)
    #
    # = Features
    #
    # * Support for any network architecture (number of layers and neurons)
    # * Configurable propagation function
    # * Optional usage of bias
    # * Configurable momentum
    # * Configurable learning rate
    # * Configurable initial weight function
    # * 100% Crystal code, no external dependency
    #
    # = Parameters
    #
    # Use class method get_parameters_info to obtain details on the algorithm
    # parameters. Use set_parameters to set values for this parameters.
    #
    # * :disable_bias => If true, the algorithm will not use bias nodes.
    #   False by default.
    # * :initial_weight_function => f(n, i, j) must return the initial
    #   weight for the conection between the node i in layer n, and node j in
    #   layer n+1. By default a random number in [-1, 1) range.
    # * :propagation_function => By default:
    #   lambda { |x| 1/(1+Math.exp(-1*(x))) }
    # * :derivative_propagation_function => Derivative of the propagation
    #   function, based on propagation function output.
    #   By default: lambda { |y| y*(1-y) }, where y=propagation_function(x)
    # * :learning_rate => By default 0.25
    # * :momentum => By default 0.1. Set this parameter to 0 to disable
    #   momentum
    #
    # = How to use it
    #
    #   # Create the network with 4 inputs, 1 hidden layer with 3 neurons,
    #   # and 2 outputs
    #   net = Ai4cr::NeuralNetwork::Backpropagation.new([4, 3, 2])
    #
    #   # Train the network
    #   1000.times do |i|
    #     net.train(example[i], result[i])
    #   end
    #
    #   # Use it: Evaluate data with the trained network
    #   net.eval([12, 48, 12, 25])
    # => [0.86, 0.01]
    #
    # More about multilayer perceptron neural networks and backpropagation:
    #
    # * http://en.wikipedia.org/wiki/Backpropagation
    # * http://en.wikipedia.org/wiki/Multilayer_perceptron
    #
    # = About the project
    # Ported By:: Daniel Huffman
    # Url::       https://github.com/drhuffman12/ai4cr
    #
    # Based on::  Ai4r
    #   Author::    Sergio Fierens
    #   License::   MPL 1.1
    #   Url::       http://ai4r.org
    #
    # Creates a new network specifying the its architecture.
    # E.g.
    #
    #   net = Backpropagation.new([4, 3, 2])  # 4 inputs
    #                                         # 1 hidden layer with 3 neurons,
    #                                         # 2 outputs
    #   net = Backpropagation.new([2, 3, 3, 4])   # 2 inputs
    #                                             # 2 hidden layer with 3 neurons each,
    #                                             # 4 outputs
    #   net = Backpropagation.new([2, 1])   # 2 inputs
    #                                       # No hidden layer
    #                                       # 1 output
    struct Backpropagation
      include Ai4cr::NeuralNetwork::Concerns::Backprop
    end
  end
end
