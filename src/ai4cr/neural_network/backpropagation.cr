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
    struct Backpropagation
      # class Backpropagation
      include ::JSON::Serializable

      # include Ai4cr::BreedParent(self.class)

      property structure, disable_bias, learning_rate, momentum
      property weights, last_changes, activation_nodes
      getter height, hidden_qty, width, deltas

      property expected_outputs : Array(Float64)

      getter error_stats : Ai4cr::ErrorStats

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

      @activation_nodes : Array(Array(Float64))
      @weights : Array(Array(Array(Float64)))
      @last_changes : Array(Array(Array(Float64)))
      @deltas : Array(Array(Float64))

      def height
        @structure.first.to_i
      end

      def hidden_qty
        @structure[1..-2]
      end

      def width
        @structure.last.to_i
      end

      def initial_weight_function
        ->(_n : Int32, _i : Int32, _j : Int32) { Ai4cr::Data::Utils.rand_neg_one_to_pos_one_no_zero }
      end

      def propagation_function
        ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
      end

      def derivative_propagation_function
        ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
      end

      def learning_style
        :sigmoid
      end

      def initialize(
        @structure : Array(Int32),
        disable_bias : Bool? = nil,
        learning_rate : Float64? = nil,
        momentum : Float64? = nil,
        history_size : Int32 = 10
        # name_instance = ""
      )
        # @name = init_name(name_instance)

        @disable_bias = !!disable_bias # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
        @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
        @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1

        # Below are set via #init_network, but must be initialized in the 'initialize' method to avoid being nilable:
        @activation_nodes = [[0.0]]
        @weights = [[[0.0]]]
        @last_changes = [[[0.0]]]

        @deltas = (@structure.size - 1).downto(1).to_a.map do |idl|
          (0..(@structure[idl] - 1)).to_a.map { 0.0 }
        end

        @expected_outputs = Array.new(width, 0.0)

        @error_stats = Ai4cr::ErrorStats.new(history_size)

        init_network
      end

      # Evaluates the input.
      # E.g.
      #     net = Backpropagation.new([4, 3, 2])
      #     net.eval([25, 32.3, 12.8, 1.5])
      #         # =>  [0.83, 0.03]
      def eval(input_values)
        input_values = input_values.map { |v| v.to_f }
        check_input_dimension(input_values.size)
        init_network if !@weights
        feedforward(input_values)
        @activation_nodes.last.clone
      end

      # Evaluates the input and returns most active node
      # E.g.
      #     net = Backpropagation.new([4, 3, 2])
      #     net.eval_result([25, 32.3, 12.8, 1.5])
      #         # eval gives [0.83, 0.03]
      #         # =>  0
      def eval_result(input_values)
        result = eval(input_values)
        result.index(result.max)
      end

      # This method trains the network using the backpropagation algorithm.
      #
      # input: Networks input
      #
      # output: Expected output for the given input.
      #
      # This method returns the network error:
      # => 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      def train(inputs, outputs)
        # inputs = inputs.map { |v| v.to_f }
        outputs = outputs.map { |v| v.to_f }
        eval(inputs)
        load_expected_outputs(outputs)
        backpropagate            # (outputs)
        calculate_error_distance # (outputs)
      end

      # Initialize (or reset) activation nodes and weights, with the
      # provided net structure and parameters.
      def init_network(history_size = 10)
        init_activation_nodes
        init_weights
        init_last_changes

        @error_stats = Ai4cr::ErrorStats.new(history_size)

        self
      end

      # protected

      # Propagate error backwards
      def backpropagate # (expected_outputs_values)
        # check_output_dimension(@expected_outputs.size)
        calculate_output_deltas # (@expected_outputs)
        calculate_internal_deltas
        update_weights
      end

      # Propagate values forward
      def feedforward(input_values)
        input_values.each_with_index do |_elem, input_index|
          @activation_nodes.first[input_index] = input_values[input_index]
        end
        @weights.each_with_index do |_elem, n|
          @structure[n + 1].times do |j|
            sum = 0.0
            @activation_nodes[n].each_with_index do |_elem, i|
              sum += (@activation_nodes[n][i] * @weights[n][i][j])
            end
            @activation_nodes[n + 1][j] = propagation_function.call(sum)
          end
        end
      end

      # Initialize neurons structure.
      def init_activation_nodes
        @activation_nodes = (0...@structure.size).map do |n|
          (0...@structure[n]).map { 1.0 }
        end
        if !disable_bias
          @activation_nodes[0...-1].each { |layer| layer << 1.0 }
        end
        @activation_nodes
      end

      # Initialize the weight arrays using function specified with the
      # initial_weight_function parameter
      def init_weights
        @weights = (0...@structure.size - 1).map do |i|
          nodes_origin_size = @activation_nodes[i].size
          nodes_target_size = @structure[i + 1]
          (0...nodes_origin_size).map do |j|
            (0...nodes_target_size).map do |k|
              initial_weight_function.call(i, j, k)
            end
          end
        end
      end

      # Momentum usage need to know how much a weight changed in the
      # previous training. This method initialize the @last_changes
      # structure with 0 values.
      def init_last_changes
        @last_changes = (0...@weights.size).map do |w|
          (0...@weights[w].size).map do |i|
            (0...@weights[w][i].size).map { 0.0 }
          end
        end
      end

      # Calculate deltas for output layer
      def calculate_output_deltas # (expected_values)
        output_values = @activation_nodes.last
        output_deltas = [] of Float64
        output_values.each_with_index do |_elem, output_index|
          error = @expected_outputs[output_index] - output_values[output_index]
          output_deltas << derivative_propagation_function.call(output_values[output_index]) * error
        end
        @deltas = [output_deltas]
      end

      # Calculate deltas for hidden layers
      def calculate_internal_deltas
        prev_deltas = @deltas.last
        (@activation_nodes.size - 2).downto(1) do |layer_index|
          layer_deltas = [] of Float64
          @activation_nodes[layer_index].each_with_index do |_elem, j|
            error = 0.0
            @structure[layer_index + 1].times do |k|
              error += prev_deltas[k] * @weights[layer_index][j][k]
            end
            layer_deltas << (derivative_propagation_function.call(@activation_nodes[layer_index][j]) * error)
          end
          prev_deltas = layer_deltas
          @deltas.unshift(layer_deltas)
        end
      end

      # Update weights after @deltas have been calculated.
      def update_weights
        # per layer from last to first...
        # n == layer number
        (@weights.size - 1).downto(0) do |n|
          # per input row weights from first to last...
          # i == input row number
          @weights[n].each_with_index do |_elem, i|
            # per output column weights from first to last...
            # j == out column number
            @weights[n][i].each_with_index do |_elem, j|
              change = @deltas[n][j]*@activation_nodes[n][i]
              @weights[n][i][j] += (learning_rate * change +
                                    momentum * @last_changes[n][i][j])
              @last_changes[n][i][j] = change
            end
          end
        end
      end

      def load_expected_outputs(expected_outputs)
        @expected_outputs.map_with_index! { |_, i| expected_outputs[i] }
      end

      # Calculate quadratic error for a expected output value
      # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      def calculate_error_distance # (expected_outputs)
        output_values = @activation_nodes.last
        error = 0.0
        @expected_outputs.each_with_index do |_elem, output_index|
          error += 0.5*(output_values[output_index] - @expected_outputs[output_index])**2
        end
        @error_stats.distance = error
        @error_stats.distance
      end

      def check_input_dimension(inputs)
        if inputs != @structure.first
          msg = "Wrong number of inputs. " +
                "Expected: #{@structure.first}, " +
                "received: #{inputs}."
          raise ArgumentError.new(msg)
        end
      end

      def check_output_dimension # (outputs)
        if @expected_outputs.size != @structure.last
          msg = "Wrong number of outputs. " +
                "Expected: #{@structure.last}, " +
                "received: #{@expected_outputs.size}."
          raise ArgumentError.new(msg)
        end
      end

      # GUESSES
      def guesses_best
        # guesses_as_is
        guesses_rounded
      end

      # # To get the sorted/top/bottom n output results
      def guesses_as_is
        @activation_nodes.last
      end

      def guesses_sorted
        @activation_nodes.last.map_with_index { |o, idx| [idx, o].sort }
      end

      def guesses_rounded # good for MiniNet::Sigmoid; and maybe MiniNetRanh
        @activation_nodes.last.map { |v| v.round }
      end

      def guesses_ceiled # good for MiniNetRelu
        @activation_nodes.last.map { |v| v.ceil }
      end

      def guesses_top_n(n = @activation_nodes.last.size)
        guesses_sorted[0..(n - 1)]
      end

      def guesses_bottom_n(n = @activation_nodes.last.size)
        guesses_sorted.reverse[0..(n - 1)]
      end
    end
  end
end
