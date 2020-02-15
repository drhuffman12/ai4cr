require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module  MiniNet
        abstract struct Common
          # This is a mini backprop networks; no hidden layers.
          #   Instead of hidden layers, you would string multiple MiniNet's together.
    
          # TODO: It can be run (via wrapper?) in states (e.g.: init, load, guess, calc_output_errors, calc_output_deltas, adjust_weights, calc_input_deltas, done)
    
          include JSON::Serializable
    
          getter width : Int32, height : Int32
          getter height_considering_bias : Int32
          getter range_width : Array(Int32), range_height : Array(Int32)
          property inputs_given : Array(Float64), outputs_guessed : Array(Float64)
          property weights : Array(Array(Float64))
          property last_changes : Array(Array(Float64)) # aka previous weights
          property error_total : Float64
    
          property outputs_expected : Array(Float64)
    
          property input_deltas : Array(Float64), output_deltas : Array(Float64)
    
          property disable_bias : Bool
          property learning_rate  : Float64
          property momentum : Float64
    
          property error_distance_history_max : Int32
          property error_distance_history : Array(Float64)
                                                                                                                                       
          def initialize(
            @height, @width,
            disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil,
            error_distance_history_max : Int32 = 10
          )
            @disable_bias = !!disable_bias
            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
            @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand
            
            # init_network:
            @height_considering_bias = @height + (@disable_bias ? 0 : 1)
            @range_height = Array.new(@height_considering_bias) { |i| i }
    
            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = 1 unless @disable_bias
            @input_deltas = Array.new(@height_considering_bias, 0.0)
    
            @range_width = Array.new(@width) { |i| i }
    
            @outputs_guessed = Array.new(@width, 0.0)
            @outputs_expected = Array.new(@width, 0.0)
            @output_deltas = Array.new(@width, 0.0)        
    
            @weights = @range_height.map { @range_width.map { rand*2-1 } }
    
            @last_changes = Array.new(@height_considering_bias, Array.new(@width, 0.0))
    
            @error_total = 0.0
    
            @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
            @error_distance = 0.0
            @error_distance_history = Array.new(0, 0.0)
          end
    
          def init_network(error_distance_history_max : Int32 = 10)
            # init_network:
            @height_considering_bias = @height + (@disable_bias ? 0 : 1)
            @range_height = Array.new(@height_considering_bias) { |i| i }
    
            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = 1 unless @disable_bias
            @input_deltas = Array.new(@height_considering_bias, 0.0)
    
            @range_width = Array.new(@width) { |i| i }
    
            @outputs_guessed = Array.new(@width, 0.0)
            @outputs_expected = Array.new(@width, 0.0)
            @output_deltas = Array.new(@width, 0.0)        
    
            @weights = @range_height.map { @range_width.map { rand*2-1 } }
    
            @last_changes = Array.new(@height_considering_bias, Array.new(@width, 0.0))
    
            @error_total = 0.0
    
            @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
            @error_distance = 0.0
            @error_distance_history = Array.new(0, 0.0)
          end
    
          ## steps for 'eval' aka 'guess':
          def eval(inputs_given) # aka eval
            step_load_inputs(inputs_given)
            step_calc_forward
            # ...
    
            @outputs_guessed
          end
    
          def guesses_best
            @outputs_guessed
          end
    
          ## To get the sorted/top/bottom n output results
          def guesses_sorted
            @outputs_guessed.map_with_index { |o, idx| [idx,o].sort }
          end
    
          def guesses_rounded # good for MiniNetExp; and maybe MiniNetRanh
            @outputs_guessed.map { |v| v.round }
          end
    
          def guesses_ceiled # good for MiniNetRelu
            @outputs_guessed.map { |v| v.ceil }
          end
    
          def guesses_top_n(n = @outputs_guessed.size)
            guesses_sorted[0..(n-1)]
          end
    
          def guesses_bottom_n(n = @outputs_guessed.size)
            guesses_sorted.reverse[0..(n-1)]
          end
    
          ## training steps
          def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
            step_load_inputs(inputs_given)
            step_calc_forward
            # ...
    
            step_load_outputs(outputs_expected)
            step_backpropagate
            step_calculate_error
    
            # {outputs_guessed: @outputs_guessed, deltas: @deltas, error: @error}
            @error_total # @error
          end
    
          def step_load_inputs(inputs)
            raise "Invalid inputs_given size: #{inputs.size}; should be height: #{@height}" if inputs.size != @height
            load_inputs(inputs)
          end
    
          def step_load_outputs(outputs_expected)
            raise "Invalid outputs_expected size" if outputs_expected.size != @width
            load_outputs_expected(outputs_expected)
          end
    
          # This would be a chained MiniNet's input_deltas
          # e.g.: mini_net_A feeds is chained into mini_net_B
          #    So you would mini_net_A.step_load_chained_outputs_deltas(mini_net_B.input_deltas)
          def step_load_chained_outputs_deltas(outputs_deltas)
            raise "Invalid outputs_deltas size" if outputs_expected.size != @width
            load_outputs_deltas(outputs_deltas)
          end
    
          def step_backpropagate
            step_calculate_output_deltas
            step_calc_input_deltas
            step_update_weights
          end
    
          # private
    
          def load_inputs(inputs_given)
            # Network could have a bias, which is racked onto to the end of the inputs, so we must account for that.
            inputs_given.each_with_index { |v,i| @inputs_given[i] = v.to_f }
          end
    
          def load_outputs_expected(outputs_expected)
            @outputs_expected.map_with_index! { |v, i| outputs_expected[i] }
          end
    
          def load_outputs_deltas(outputs_deltas)
            @outputs_deltas.map_with_index! { |v, i| outputs_deltas[i] }
          end
    
          ####
          # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
          def propagation_function
            ->(x : Float64) { x } # { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
          end
    
          # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
          def derivative_propagation_function()
            ->(y : Float64) { y } # { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
          end
          ####
          
          def step_calc_forward # aka feedforward # step_calc_forward_1
            # 1nd place WINNER w/ 100x100 i's and o's
    
            # close tie beteen step_calc_forward_1 and step_calc_forward_2 as fastest
            @outputs_guessed = @range_width.map do |w|
              sum = 0.0
              @range_height.each do |h|
                sum += @inputs_given[h]*@weights[h][w]
              end
              propagation_function.call(sum)
              # sum
            end
          end
    
          # Calculate deltas for output layer
          def step_calculate_output_deltas # (outputs_expected)
            @output_deltas.map_with_index! do |d, i|
              error = @outputs_expected[i] - @outputs_guessed[i]
              derivative_propagation_function.call(@outputs_guessed[i]) * error
            end
          end
    
          # Calculate deltas for hidden layers
          def step_calc_input_deltas # calculate_internal_deltas
            prev_deltas = @output_deltas
            layer_index = 1
            layer_deltas = [] of Float64
            height_considering_bias.times.to_a.each do |j|
              error = 0.0
              @width.times do |k|
                error += @output_deltas[k] * @weights[j][k]
              end
              layer_deltas << (derivative_propagation_function.call(@inputs_given[j]) * error)
            end
            @input_deltas = layer_deltas
          end
    
          # Update weights after @deltas have been calculated.
          def step_update_weights # update_weights
            # per input row weights from first to last...
            # j == input row number
            height_considering_bias.times.to_a.each do |j|
              # per output column weights from first to last...
              # k == out column number
              @weights[j].each_with_index do |_elem, k|
                change = @output_deltas[k]*@inputs_given[j]
                @weights[j][k] += (learning_rate * change + momentum * @last_changes[j][k])
                @last_changes[j][k] = change
              end
            end
          end
    
          def step_calculate_error # aka calculate_error
            error = 0.0
            @outputs_expected.map_with_index do |oe, iw|
              error += 0.5*(oe - @outputs_guessed[iw])**2
            end
            @error_total = error
          end
    
          # Calculate the radius of the error as if each output cell is an value in a coordinate set
          def step_calculate_error_distance_history
            # @error_distance_history_max = error_distance_history_max
            return @error_distance_history = [-1.0] if @error_distance_history_max < 1
            error = 0.0
            @outputs_expected.map_with_index do |oe, iw|
              error += (oe - @outputs_guessed[iw])**2
            end
            @error_distance = Math.sqrt(error)
            if @error_distance_history.size < @error_distance_history_max - 1
              # Array not 'full' yet, so add latest value to end
              @error_distance_history << error_total
            else
              # Array 'full', so rotate end to front and then put new value at last index
              @error_distance_history.rotate!
              @error_distance_history[-1] = error_total
            end
            @error_distance_history
          end
        end
      end
    end
  end
end

# puts Ai4cr::NeuralNetwork::Cmn::MiniNet::.new(2,3).to_json
