module Ai4cr
  module NeuralNetwork
    module Backpropagation
      struct Net # Trainer
        include JSON::Serializable
        include Ai4cr::NeuralNetwork::Backpropagation::Math

        property state : State
        # property train_till_error_below : Float64

        # property error_averages : Array(Float64)

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

        def initialize(structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil) # , @train_till_error_below = 0.1)
          @state = State.new(structure, disable_bias, learning_rate, momentum)
        end

        def training_stats(label = "Net Training Stats:", min : Float64 = 0.0, max : Float64 = 1.0, precision = 3.to_i8, in_bw = false)
          charter = AsciiBarCharter.new(min, max, precision, in_bw)

          main_data = {
            label: label,
            config: JSON.parse(state.config.to_json),
            learning_rate: state.config.learning_rate.round(precision),
            calculated_error_latest:  state.calculated_error_latest,
            bar_last: charter.bar_prefixed_with_number(state.calculated_error_latest),
            track_history:  state.track_history,
          }

          unless (state.track_history && state.calculated_error_history.size > 0)
            main_data
          else
            # raise "You made it! :)"
            main_data.merge({
              calculated_error_history_size:  state.calculated_error_history.size,
              calculated_error_history:  state.calculated_error_history,
              bar_first: charter.bar_prefixed_with_number(state.calculated_error_history.first),
              bars: charter.plot(state.calculated_error_history)
              # error_averages: 
            })
          end
        end
        
        # # TODO (?): remove config alias
        # def config
        #   state.config
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
            structure:        state.config.structure,
            disable_bias:     state.config.disable_bias,
            learning_rate:    state.config.learning_rate,
            momentum:         state.config.momentum,
            weights:          state.weights,
            last_changes:     state.last_changes,
            activation_nodes: state.activation_nodes,
          }
        end

        @[Deprecated("Use `self.from_json(json_data)` instead")]
        def marshal_load(tup)
          state.config.structure = tup[:structure].as(Array(Int32))
          state.config.disable_bias = tup[:disable_bias].as(Bool)
          state.config.learning_rate = tup[:learning_rate].as(Float64)
          state.config.momentum = tup[:momentum].as(Float64)
          state.weights = tup[:weights].as(Array(Array(Array(Float64))))
          state.last_changes = tup[:last_changes].as(Array(Array(Array(Float64))))
          state.activation_nodes = tup[:activation_nodes].as(Array(Array(Float64)))
          # @initial_weight_function = lambda { |n, i, j| ((rand(2000))/1000.0) - 1}
          # @propagation_function = lambda { |x| 1/(1+Math.exp(-1*(x))) } #lambda { |x| Math.tanh(x) }
          # @derivative_propagation_function = lambda { |y| y*(1-y) } #lambda { |y| 1.0 - y**2 }
        end

        # NOTE: To make a net of similar config, use config_* instead of *_json
        @[Deprecated("Use `self.to_json` instead")]
        def to_config
          state.config.to_json
        end

        @[Deprecated("Use `self.from_json(json_data)` instead")]
        def self.from_config(json)
          state_config = Ai4cr::NeuralNetwork::Backpropagation::Config.from_json(json)

          Ai4cr::NeuralNetwork::Backpropagation::Net.new(
            structure: state_config.structure,
            disable_bias: state_config.disable_bias,
            learning_rate: state_config.learning_rate,
            momentum: state_config.momentum,
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
        #     net = Backpropagation.new([4, 3, 3])
        #     net.eval_result([25, 32.3, 12.8, 1.5])
        #         # eval gives [0.83, 0.83, 0.03]
        #         # =>  0
        def eval_result(input_values)
          result = eval(input_values)
          result.index(result.max) # tie goes to the lowest index
          # results.map_with_index{|result, index| [result, index]}.sort.reverse.first[1] # tie goes to the highest index
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
          if outputs != state.config.output_size
            msg = "Wrong number of outputs. " +
                  "Expected: #{state.config.output_size}, " +
                  "received: #{outputs}."
            raise ArgumentError.new(msg)
          end
        end

        def adjusted_inputs
          using_nodes = state.activation_nodes.first[0..(state.config.input_size - 1)]
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
              state.config.structure[layer_index + 1].times do |k|
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
                state.weights[n][i][j] += (state.config.learning_rate * change +
                                      state.config.momentum * state.last_changes[n][i][j])
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
          state.update_calculated_error_latest(error)
        end
        
        ################################
        ## Forward (eval):

        private def check_input_dimension(inputs)
          if inputs != state.config.input_size
            msg = "Wrong number of inputs. " +
                  "Expected: #{state.config.input_size}, " +
                  "received: #{inputs}."
            raise ArgumentError.new(msg)
          end
        end

        private def feedforward(input_values)
          input_values.each_with_index do |_elem, input_index|
            state.activation_nodes.first[input_index] = input_values[input_index]
          end
          state.weights.each_with_index do |_elem, n|
            state.config.structure[n + 1].times do |j|
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