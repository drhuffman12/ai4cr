module Ai4cr
  module NeuralNetwork
    module Backpropagation
      struct Net
        include JSON::Serializable
        include Ai4cr::NeuralNetwork::Backpropagation::Math

        # property config : Config
        # getter height, hidden_qty, width

        property state : State
        # property weights, last_changes, activation_nodes
        # property deltas, input_deltas

        # property calculated_error_total : Float64

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

        # state.activation_nodes : Array(Array(Float64))
        # state.weights : Array(Array(Array(Float64)))
        # state.last_changes : Array(Array(Array(Float64)))
        # @deltas : Array(Array(Float64))
        # @input_deltas : Array(Float64)

        def initialize(structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
          # state.config = Config.new(structure, disable_bias, learning_rate, momentum)
          # @state = State.new(state.config)
          @state = State.new(structure, disable_bias, learning_rate, momentum)

          # config.disable_bias = !!disable_bias
          # config.learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
          # config.momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1

          # state.activation_nodes = init_activation_nodes
          # state.weights = init_weights
          # state.last_changes = init_last_changes
          # state.deltas = init_deltas
          # state.input_deltas = init_input_deltas
          # state.calculated_error_total = 0.0
        end

        # TODO (?): remove config alias
        def config
          state.config
        end

        # def height
        #   config.height
        # end

        # def hidden_qty
        #   config.hidden_layer_sizes
        # end

        # def width
        #   config.width
        # end

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
            structure:        config.structure,
            disable_bias:     config.disable_bias,
            learning_rate:    config.learning_rate,
            momentum:         config.momentum,
            weights:          state.weights,
            last_changes:     state.last_changes,
            activation_nodes: state.activation_nodes,
          }
        end

        @[Deprecated("Use `self.from_json(json_data)` instead")]
        def marshal_load(tup)
          config.structure = tup[:structure].as(Array(Int32))
          config.disable_bias = tup[:disable_bias].as(Bool)
          config.learning_rate = tup[:learning_rate].as(Float64)
          config.momentum = tup[:momentum].as(Float64)
          state.weights = tup[:weights].as(Array(Array(Array(Float64))))
          state.last_changes = tup[:last_changes].as(Array(Array(Array(Float64))))
          state.activation_nodes = tup[:activation_nodes].as(Array(Array(Float64)))
          # @initial_weight_function = lambda { |n, i, j| ((rand(2000))/1000.0) - 1}
          # @propagation_function = lambda { |x| 1/(1+Math.exp(-1*(x))) } #lambda { |x| Math.tanh(x) }
          # @derivative_propagation_function = lambda { |y| y*(1-y) } #lambda { |y| 1.0 - y**2 }
        end

        # NOTE: To make a net of similar config, use config_* instead of *_json
        def to_config
          # {
          #   structure:        config.structure,
          #   disable_bias:     config.disable_bias,
          #   learning_rate:    config.learning_rate,
          #   momentum:         config.momentum,
          # }.to_json
          state.config.to_json
        end

        def self.from_config(json)
          config = Ai4cr::NeuralNetwork::Backpropagation::Config.from_json(json)

          # structure = config.structure
          # disable_bias = config.disable_bias
          # learning_rate = config.learning_rate
          # momentum = config.momentum

          Ai4cr::NeuralNetwork::Backpropagation::Net.new(
            structure: config.structure,
            disable_bias: config.disable_bias,
            learning_rate: config.learning_rate,
            momentum: config.momentum,
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
          outputs = state.activation_nodes.last.map { |v| v.to_f }
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
          # init_network if !state.weights
          feedforward(input_values)
          return state.activation_nodes.last.clone
        end

        ################################ private ################################

        # ################################
        # ## moved to Ai4cr::NeuralNetwork::Backpropagation::Math

        # def initial_weight_function
        #   ->(n : Int32, i : Int32, j : Int32) { ((rand(2000))/1000.0) - 1 }
        # end

        # def propagation_function
        #   ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
        # end

        # def derivative_propagation_function
        #   ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
        # end

        ################################
        ## Initialization .. moved to Ai4cr::NeuralNetwork::Backpropagation::State

        # # TODO: Remove
        # # Initialize (or reset) activation nodes and weights, with the
        # # provided net structure and parameters.
        # def init_network
        #   init_activation_nodes
        #   init_weights
        #   init_last_changes
        #   init_deltas
        #   return self
        # end

        # # Initialize neurons structure.
        # private def init_activation_nodes
        #   act_nodes = (0...config.structure.size).map do |n|
        #     (0...config.structure[n]).map { 1.0 }
        #   end
        #   if !config.disable_bias
        #     act_nodes[0...-1].each { |layer| layer << 1.0 }
        #   end
        #   act_nodes
        # end

        # # Initialize the weight arrays using function specified with the
        # # initial_weight_function parameter
        # private def init_weights
        #   (0...config.structure.size - 1).map do |i|
        #     nodes_origin_size = state.activation_nodes[i].size
        #     nodes_target_size = config.structure[i + 1]
        #     (0...nodes_origin_size).map do |j|
        #       (0...nodes_target_size).map do |k|
        #         initial_weight_function.call(i, j, k)
        #       end
        #     end
        #   end
        # end

        # # Momentum usage need to know how much a weight changed in the
        # # previous training. This method initialize the state.last_changes
        # # structure with 0 values.
        # private def init_last_changes
        #   (0...state.weights.size).map do |w|
        #     (0...state.weights[w].size).map do |i|
        #       (0...state.weights[w][i].size).map { 0.0 }
        #     end
        #   end
        # end

        # private def init_deltas
        #   config.structure.map{|layer_size| layer_size.times.map{0.0}.to_a}.to_a
        # end
        
        # private def init_input_deltas
        #   config.structure.first.times.map{0.0}.to_a
        # end

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
          if outputs != config.structure.last
            msg = "Wrong number of outputs. " +
                  "Expected: #{config.structure.last}, " +
                  "received: #{outputs}."
            raise ArgumentError.new(msg)
          end
        end

        def adjusted_inputs
          using_nodes = state.activation_nodes.first[0..(config.structure.first - 1)]
          using_delta = state.deltas.first # last

          raise "The adjusted_inputs sizes don't match! using_nodes.size: #{using_nodes.size}, using_delta.size: #{using_delta.size}" if using_nodes.size != using_delta.size

          using_nodes.map_with_index do |val,i|
            val + using_delta[i]
          end
        end

        ## For backprop of chained networks

        # Calculate deltas for output layer
        private def load_output_deltas(other_net_deltas_last)
          state.deltas = [other_net_deltas_last]
        end

        ## For backprop of end-of-chained networks
        # Calculate deltas for output layer
        private def calculate_output_deltas(expected_values)
          output_values = state.activation_nodes.last
          output_deltas = [] of Float64
          output_values.each_with_index do |_elem, output_index|
            error = expected_values[output_index] - output_values[output_index]
            output_deltas << derivative_propagation_function.call(output_values[output_index]) * error
          end
          state.deltas = [output_deltas]
        end

        # Calculate deltas for hidden layers
        private def calculate_internal_deltas
          prev_deltas = state.deltas.last
          (state.activation_nodes.size - 2).downto(0) do |layer_index|
            layer_deltas = [] of Float64
            state.activation_nodes[layer_index].each_with_index do |_elem, j|
              error = 0.0
              config.structure[layer_index + 1].times do |k|
                error += prev_deltas[k] * state.weights[layer_index][j][k]
              end
              layer_deltas << (derivative_propagation_function.call(state.activation_nodes[layer_index][j]) * error)
            end
            prev_deltas = layer_deltas
            if layer_index == 0
              state.input_deltas = layer_deltas # to pass back to prior chained nets
            else
              state.deltas.unshift(layer_deltas) # for current net
            end
          end
        end

        # Update weights after @deltas have been calculated.
        private def update_weights
          (state.weights.size - 1).downto(0) do |n|
            state.weights[n].each_with_index do |_elem, i|
              state.weights[n][i].each_with_index do |_elem, j|
                change = state.deltas[n][j]*state.activation_nodes[n][i]
                state.weights[n][i][j] += (config.learning_rate * change +
                                      config.momentum * state.last_changes[n][i][j])
                state.last_changes[n][i][j] = change
              end
            end
          end
        end

        # Calculate quadratic error for a expected output value
        # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
        private def calculate_error(expected_output)
          output_values = state.activation_nodes.last
          error = 0.0
          expected_output.each_with_index do |_elem, output_index|
            error += 0.5*(output_values[output_index] - expected_output[output_index])**2
          end
          state.calculated_error_total = error
        end
        
        ################################
        ## Forward (eval):

        private def check_input_dimension(inputs)
          if inputs != config.structure.first
            msg = "Wrong number of inputs. " +
                  "Expected: #{config.structure.first}, " +
                  "received: #{inputs}."
            raise ArgumentError.new(msg)
          end
        end

        private def feedforward(input_values)
          input_values.each_with_index do |_elem, input_index|
            state.activation_nodes.first[input_index] = input_values[input_index]
          end
          state.weights.each_with_index do |_elem, n|
            config.structure[n + 1].times do |j|
              sum = 0.0
              state.activation_nodes[n].each_with_index do |_elem, i|
                sum += (state.activation_nodes[n][i] * state.weights[n][i][j])
              end
              state.activation_nodes[n + 1][j] = propagation_function.call(sum)
            end
          end
        end
      end
    end
  end
end