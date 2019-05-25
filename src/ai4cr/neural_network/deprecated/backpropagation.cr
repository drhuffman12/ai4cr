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
require "json"

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
    #   net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([4, 3, 2])
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
      include ::JSON::Serializable
      
      property structure, disable_bias, learning_rate, momentum
      property weights, last_changes, activation_nodes
      property calculated_error_total : Float64
      property deltas, input_deltas
      getter height, hidden_qty, width

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
      @input_deltas : Array(Float64)

      def initialize(@structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
        @disable_bias = !!disable_bias
        @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
        @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1

        @activation_nodes = init_activation_nodes
        @weights = init_weights
        @last_changes = init_last_changes
        @deltas = init_deltas
        @input_deltas = init_input_deltas
        @calculated_error_total = 0.0
      end

      def height
        @structure.first.to_i
      end

      def hidden_qty
        @structure[1..-2]
      end

      def width
        @structure.last.to_i
      end

      ################################
      ## Loading and Saving methods:

      # Custom serialization. It used to fail trying to serialize because
      # it uses lambda functions internally, and they cannot be serialized.
      # Now it does not fail, but if you customize the values of
      # * initial_weight_function
      # * propagation_function
      # * derivative_propagation_function
      # you must restore their values manually after loading the instance.
      @[Deprecated("Use `self.to_json` instead")]
      def marshal_dump
        {
          structure:        @structure,
          disable_bias:     @disable_bias,
          learning_rate:    @learning_rate,
          momentum:         @momentum,
          weights:          @weights,
          last_changes:     @last_changes,
          activation_nodes: @activation_nodes,
        }
      end

      @[Deprecated("Use `self.from_json(json_data)` instead")]
      def marshal_load(tup)
        @structure = tup[:structure].as(Array(Int32))
        @disable_bias = tup[:disable_bias].as(Bool)
        @learning_rate = tup[:learning_rate].as(Float64)
        @momentum = tup[:momentum].as(Float64)
        @weights = tup[:weights].as(Array(Array(Array(Float64))))
        @last_changes = tup[:last_changes].as(Array(Array(Array(Float64))))
        @activation_nodes = tup[:activation_nodes].as(Array(Array(Float64)))
        # @initial_weight_function = lambda { |n, i, j| ((rand(2000))/1000.0) - 1}
        # @propagation_function = lambda { |x| 1/(1+Math.exp(-1*(x))) } #lambda { |x| Math.tanh(x) }
        # @derivative_propagation_function = lambda { |y| y*(1-y) } #lambda { |y| 1.0 - y**2 }
      end

      # NOTE: To make a net of similar config, use config_* instead of *_json
      def to_config
        {
          structure:        @structure,
          disable_bias:     @disable_bias,
          learning_rate:    @learning_rate,
          momentum:         @momentum,
        }.to_json
      end

      def self.from_config(json)
        config = Ai4cr::NeuralNetwork::BackpropagationConfig.from_json(json)

        structure = config.structure
        disable_bias = config.disable_bias
        learning_rate = config.learning_rate
        momentum = config.momentum

        Ai4cr::NeuralNetwork::Backpropagation::Net.new(
          structure: structure,
          disable_bias: disable_bias,
          learning_rate: learning_rate,
          momentum: momentum,
        )
      end

      ################################################################
      ## Train and Eval

      # This method trains the network using the backpropagation algorithm.
      #
      # input: Networks input
      #
      # output: Expected output for the given input.
      #
      # This method returns the network error:
      # => 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      def train(inputs, outputs)
        train_forward(inputs)
        train_backwards(outputs)
      end

      def train_from_chained_net2(inputs, net2_deltas_last) # net2_adjusted_inputs
        train_forward(inputs)
        train_backwards_from_chained_net(net2_deltas_last)
      end

      def train_forward(inputs)
        eval(inputs)
      end

      def train_backwards(outputs)
        outputs = outputs.map { |v| v.to_f }
        backpropagate(outputs)
        calculate_error(outputs)
      end

      # def train_backwards_from_chained_net(net2_adjusted_inputs, net2_deltas_last)
      def train_backwards_from_chained_net(net2_deltas_last) # net2_inputs, # activation_nodes.last.clone, 
        outputs = activation_nodes.last.map { |v| v.to_f }
        backpropagate_from_chained_net(outputs, net2_deltas_last)
        calculate_error(outputs)
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

      # Evaluates the input.
      # E.g.
      #     net = Backpropagation.new([4, 3, 2])
      #     net.eval([25, 32.3, 12.8, 1.5])
      #         # =>  [0.83, 0.03]
      def eval(input_values)
        input_values = input_values.map { |v| v.to_f }
        check_input_dimension(input_values.size)
        # init_network if !@weights
        feedforward(input_values)
        return @activation_nodes.last.clone
      end

      ################################ private ################################

      ################################
      ## Math

      def initial_weight_function
        ->(n : Int32, i : Int32, j : Int32) { ((rand(2000))/1000.0) - 1 }
      end

      def propagation_function
        ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
      end

      def derivative_propagation_function
        ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
      end

      ################################
      ## Initialization

      # TODO: Remove
      # Initialize (or reset) activation nodes and weights, with the
      # provided net structure and parameters.
      def init_network
        init_activation_nodes
        init_weights
        init_last_changes
        init_deltas
        return self
      end

      # Initialize neurons structure.
      private def init_activation_nodes
        act_nodes = (0...@structure.size).map do |n|
          (0...@structure[n]).map { 1.0 }
        end
        if !disable_bias
          act_nodes[0...-1].each { |layer| layer << 1.0 }
        end
        act_nodes
      end

      # Initialize the weight arrays using function specified with the
      # initial_weight_function parameter
      private def init_weights
        (0...@structure.size - 1).map do |i|
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
      private def init_last_changes
        (0...@weights.size).map do |w|
          (0...@weights[w].size).map do |i|
            (0...@weights[w][i].size).map { 0.0 }
          end
        end
      end

      private def init_deltas
        structure.map{|layer_size| layer_size.times.map{0.0}.to_a}.to_a
      end
      
      private def init_input_deltas
        structure.first.times.map{0.0}.to_a
      end

      ################################
      ## Backward (train):

      # Propagate error backwards from chained network
      private def backpropagate_from_chained_net(outputs, other_net_deltas_last) # net2_adjusted_inputs
        load_output_deltas(other_net_deltas_last)
        calculate_internal_deltas
        update_weights
      end

      # Propagate error backwards
      private def backpropagate(expected_output_values)
        check_output_dimension(expected_output_values.size)
        calculate_output_deltas(expected_output_values)
        calculate_internal_deltas
        update_weights
      end

      private def check_output_dimension(outputs)
        if outputs != @structure.last
          msg = "Wrong number of outputs. " +
                "Expected: #{@structure.last}, " +
                "received: #{outputs}."
          raise ArgumentError.new(msg)
        end
      end

      def adjusted_inputs
        using_nodes = @activation_nodes.first[0..(structure.first - 1)]
        using_delta = @deltas.first # last

        raise "The adjusted_inputs sizes don't match! using_nodes.size: #{using_nodes.size}, using_delta.size: #{using_delta.size}" if using_nodes.size != using_delta.size

        using_nodes.map_with_index do |val,i|
          val + using_delta[i]
        end
      end

      ## For backprop of chained networks

      # Calculate deltas for output layer
      private def load_output_deltas(other_net_deltas_last)
        @deltas = [other_net_deltas_last]
      end

      ## For backprop of end-of-chained networks
      # Calculate deltas for output layer
      private def calculate_output_deltas(expected_values)
        output_values = @activation_nodes.last
        output_deltas = [] of Float64
        output_values.each_with_index do |_elem, output_index|
          error = expected_values[output_index] - output_values[output_index]
          output_deltas << derivative_propagation_function.call(output_values[output_index]) * error
        end
        @deltas = [output_deltas]
      end

      # Calculate deltas for hidden layers
      private def calculate_internal_deltas
        prev_deltas = @deltas.last
        (@activation_nodes.size - 2).downto(0) do |layer_index|
          layer_deltas = [] of Float64
          @activation_nodes[layer_index].each_with_index do |_elem, j|
            error = 0.0
            @structure[layer_index + 1].times do |k|
              error += prev_deltas[k] * @weights[layer_index][j][k]
            end
            layer_deltas << (derivative_propagation_function.call(@activation_nodes[layer_index][j]) * error)
          end
          prev_deltas = layer_deltas
          if layer_index == 0
            @input_deltas = layer_deltas # to pass back to prior chained nets
          else
            @deltas.unshift(layer_deltas) # for current net
          end
        end
      end

      # Update weights after @deltas have been calculated.
      private def update_weights
        (@weights.size - 1).downto(0) do |n|
          @weights[n].each_with_index do |_elem, i|
            @weights[n][i].each_with_index do |_elem, j|
              change = @deltas[n][j]*@activation_nodes[n][i]
              @weights[n][i][j] += (learning_rate * change +
                                    momentum * @last_changes[n][i][j])
              @last_changes[n][i][j] = change
            end
          end
        end
      end

      # Calculate quadratic error for a expected output value
      # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      private def calculate_error(expected_output)
        output_values = @activation_nodes.last
        error = 0.0
        expected_output.each_with_index do |_elem, output_index|
          error += 0.5*(output_values[output_index] - expected_output[output_index])**2
        end
        @calculated_error_total = error
      end
      
      ################################
      ## Forward (eval):

      private def check_input_dimension(inputs)
        if inputs != @structure.first
          msg = "Wrong number of inputs. " +
                "Expected: #{@structure.first}, " +
                "received: #{inputs}."
          raise ArgumentError.new(msg)
        end
      end

      private def feedforward(input_values)
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
    end
  end
end
