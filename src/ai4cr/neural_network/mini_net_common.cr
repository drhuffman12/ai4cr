require "json"

module Ai4cr
  module NeuralNetwork
    abstract struct MiniNetCommon
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
      property calculated_error_total : Float64

      # property inputs_given : Array(Float64), 
      property outputs_expected : Array(Float64)

      property inputs_deltas : Array(Float64), output_deltas : Array(Float64)

      property disable_bias : Bool
      property learning_rate  : Float64
      property momentum : Float64

      def initialize(
        @height, @width,
        disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil
      )
        @disable_bias = !!disable_bias
        @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
        @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

        @height_considering_bias = @height + (@disable_bias ? 0 : 1)
        @range_height = Array.new(@height_considering_bias) { |i| i }

        @inputs_given = Array.new(@height_considering_bias, 0.0)
        @inputs_given[-1] = 1 unless @disable_bias
        @inputs_deltas = Array.new(@height_considering_bias, 0.0)

        @range_width = Array.new(@width) { |i| i }

        @outputs_guessed = Array.new(@width, 0.0)
        @outputs_expected = Array.new(@width, 0.0)
        @output_deltas = Array.new(@width, 0.0)        

        @weights = @range_height.map { @range_width.map { rand*2-1 } }

        @last_changes = Array.new(@height_considering_bias, Array.new(@width, 0.0))

        @calculated_error_total = 0.0
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
      def train(inputs_given, outputs_expected)
        step_load_inputs(inputs_given)
        step_calc_forward
        # ...

        step_load_outputs(outputs_expected)
        step_backpropagate
        step_calculate_error

        # {outputs_guessed: @outputs_guessed, deltas: @deltas, error: @error}
        @calculated_error_total # @error
      end

      def step_load_inputs(inputs_given)
        raise "Invalid inputs_given size: #{inputs_given.size}; should be height: #{@height}" if inputs_given.size != @height
        load_inputs(inputs_given)
      end

      def step_load_outputs(outputs_expected)
        raise "Invalid outputs_expected size" if outputs_expected.size != @width
        load_outputs_expected(outputs_expected)
      end

      # This would be a chained MiniNet's inputs_deltas
      # e.g.: mini_net_A feeds is chained into mini_net_B
      #    So you would mini_net_A.step_load_chained_outputs_deltas(mini_net_B.inputs_deltas)
      def step_load_chained_outputs_deltas(outputs_deltas)
        raise "Invalid outputs_deltas size" if outputs_expected.size != @width
        load_outputs_deltas(outputs_deltas)
      end

      def step_backpropagate
        step_calculate_output_deltas
        step_calc_input_deltas
        step_update_weights
      end

      # def step_calculate_output_deltas
      #   @outputs_expected.map_with_index do |oe, index|
      
      #     @output_deltas_raw[index] = oe - @outputs_guessed[index]
      #     @output_errors[index] = oe - @outputs_guessed[index]

      #     @output_deltas_capped[index]
      #     output_deltas << derivative_propagation_function.call(output_values[output_index]) * error
      #   end
      # end

      # private

      def load_inputs(inputs_given)
        # Network could have a bias, which is racked onto to the end of the inputs, so we must account for that.
        # @inputs_given = inputs_given.map { |v| v.to_f }}
        # @inputs_given.map_with_index! { |v, i| inputs_given[i].to_f }
        inputs_given.each_with_index { |v,i| @inputs_given[i] = v.to_f }
      end

      def load_outputs_expected(outputs_expected)
        # @outputs_expected = outputs_expected.map { |v| v.to_f }
        @outputs_expected.map_with_index! { |v, i| outputs_expected[i] }
      end

      def load_outputs_deltas(outputs_deltas)
        # @outputs_deltas = outputs_deltas.map { |v| v.to_f }
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

      def step_calculate_output_deltas # (outputs_expected)
        # step_load_outputs(outputs_expected)
        @output_deltas.map_with_index! do |d, i|
          error = @outputs_expected[i] - @outputs_guessed[i]
          derivative_propagation_function.call(@outputs_guessed[i]) * error
        end
      end

      def step_calc_input_deltas # aka calculate_internal_deltas aka step_calculate_internal_deltas
        @inputs_deltas.map_with_index! do |vi, ih|
          error = 0.0
          @range_width.map do |iw|
            # error += @inputs_given[ih]*@weights[ih][iw]
            error += @output_deltas[iw] * @weights[ih][iw]
          end
          # propagation_function.call(sum)
          derivative_propagation_function.call(@inputs_given[ih]) * error
          # sum
        end
      end

      def step_update_weights # aka update_weights
        @range_height.each do |ih|
          @range_width.each do |iw|
            # rand*2-1

            # change = @deltas[n][j]*@activation_nodes[n][i]
            # @last_changes[n][i][j] = change
            # @weights[n][i][j] += (learning_rate * change +
            #                       momentum * @last_changes[n][i][j])

            change = @output_deltas[iw] * @inputs_given[ih]
            @weights[ih][iw] = @learning_rate * change + @momentum * @last_changes[ih][iw] 
            @last_changes[ih][iw] = change           
          end
        end
      end

      def step_calculate_error # aka calculate_error
        @calculated_error_total = 0.0
        @outputs_expected.map_with_index do |oe, iw|
          @calculated_error_total += 0.5*(oe - @outputs_guessed[iw])**2
        end
        @calculated_error_total
      end
    end
  end
end

# puts Ai4cr::NeuralNetwork::MiniNet.new(2,3).to_json
