# Ai4cr::NeuralNetwork::Concerns::Backprop
module Ai4cr
  module NeuralNetwork
    module Concerns
      module Backprop
        property structure, weights, activation_nodes, last_changes
        property disable_bias, learning_rate, momentum, activation_nodes
        property height, hidden_qty, width

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

        def deltas
          @structure.last.to_i
        end

        def initial_weight_function
          ->(n : Int32, i : Int32, j : Int32) { ((rand(2000))/1000.0) - 1 }
        end

        def propagation_function
          ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
        end

        def derivative_propagation_function
          ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
        end

        def initialize(@structure : Array(Int32), disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil)
          @disable_bias = !!disable_bias
          @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
          @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1
          # Below are set via #init_network, but must be initialized in the 'initialize' method to avoid being nilable:
          @activation_nodes = [[0.0]]
          @weights = [[[0.0]]]
          @last_changes = [[[0.0]]]
          @deltas = [[0.0]]
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
          return @activation_nodes.last.clone
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
          backpropagate(outputs)
          calculate_error(outputs)
        end

        # Initialize (or reset) activation nodes and weights, with the
        # provided net structure and parameters.
        def init_network
          init_activation_nodes
          init_weights
          init_last_changes
          return self
        end

        # # protected

        # Custom serialization. It used to fail trying to serialize because
        # it uses lambda functions internally, and they cannot be serialized.
        # Now it does not fail, but if you customize the values of
        # * initial_weight_function
        # * propagation_function
        # * derivative_propagation_function
        # you must restore their values manually after loading the instance.
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

        # Propagate error backwards
        def backpropagate(expected_output_values)
          check_output_dimension(expected_output_values.size)
          calculate_output_deltas(expected_output_values)
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
        def calculate_output_deltas(expected_values)
          output_values = @activation_nodes.last
          output_deltas = [] of Float64
          output_values.each_with_index do |_elem, output_index|
            error = expected_values[output_index] - output_values[output_index]
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
        def calculate_error(expected_output)
          output_values = @activation_nodes.last
          error = 0.0
          expected_output.each_with_index do |_elem, output_index|
            error += 0.5*(output_values[output_index] - expected_output[output_index])**2
          end
          return error
        end
        
        def check_input_dimension(inputs)
          if inputs != @structure.first
            msg = "Wrong number of inputs. " +
                  "Expected: #{@structure.first}, " +
                  "received: #{inputs}."
            raise ArgumentError.new(msg)
          end
        end

        def check_output_dimension(outputs)
          if outputs != @structure.last
            msg = "Wrong number of outputs. " +
                  "Expected: #{@structure.last}, " +
                  "received: #{outputs}."
            raise ArgumentError.new(msg)
          end
        end
      end
    end
  end
end
