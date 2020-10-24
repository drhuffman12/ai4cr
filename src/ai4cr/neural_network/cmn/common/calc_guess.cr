require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module Common
        module CalcGuess
          # ####
          # # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
          # def propagation_function
          #   ->(x : Float64) { x } # { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
          # end
          abstract def propagation_function

          # ####

          # pseudo-abstract
          # default set below, but might be different per subclass
          def guesses_best
            guesses_as_is
          end

          # # To get the sorted/top/bottom n output results
          def guesses_as_is
            @outputs_guessed
          end

          def guesses_sorted
            @outputs_guessed.map_with_index { |o, idx| [idx, o].sort }
          end

          def guesses_rounded # good for MiniNet::Sigmoid; and maybe MiniNetRanh
            @outputs_guessed.map { |v| v.round }
          end

          def guesses_ceiled # good for MiniNetRelu
            @outputs_guessed.map { |v| v.ceil }
          end

          def guesses_top_n(n = @outputs_guessed.size)
            guesses_sorted[0..(n - 1)]
          end

          def guesses_bottom_n(n = @outputs_guessed.size)
            guesses_sorted.reverse[0..(n - 1)]
          end

          # steps for 'eval' aka 'guess':
          def eval(inputs_given) # aka eval
            step_load_inputs(inputs_given)
            step_calc_forward
            # ...

            @outputs_guessed
          end

          def validate_inputs(inputs, height_expected)
            # if inputs.size != height_expected
            #   raise "Invalid inputs given size: #{inputs.size}; should be height_expected: #{height_expected}"
            # end
            if (inputs.size > height_expected)
              raise "Invalid inputs given size: #{inputs.size}; should be no more than height_expected: #{height_expected}"
            end
          end

          def validate_outputs(outputs, width_expected)
            if outputs.size != width_expected
              raise "Invalid outputs size: #{outputs.size}; should be width_expected: #{width_expected}"
            end
            # if (outputs.size > width_expected)
            #   raise "Invalid outputs given size: #{outputs.size}; should be no more than width_expected: #{width_expected}"
            # end
          end

          def step_load_inputs(inputs_given)
            # raise "Invalid inputs_given size: #{inputs.size}; should be height: #{@height}" if inputs.size != @height
            validate_inputs(inputs_given, @height_considering_bias)
            load_inputs(inputs_given)
          end

          def load_inputs(inputs_given)
            # Network could have a bias, which is racked onto to the end of the inputs, so we must account for that.
            validate_inputs(inputs_given, @height_considering_bias)
            inputs_given.each_with_index { |v, i| @inputs_given[i] = v.to_f }
          end

          def step_calc_forward # aka feedforward # step_calc_forward_1
            # 1nd place WINNER w/ 100x100 i's and o's

            # puts "validate_inputs(@inputs_given, @height_considering_bias)"
            validate_inputs(@inputs_given, @height_considering_bias)

            # puts "validate_inputs(@inputs_given, @height_indexes.size)"
            # validate_inputs(@inputs_given, @height_indexes.size)

            # puts "validate_outputs(@inputs_given, @height_considering_bias)"
            validate_outputs(@outputs_guessed, @width)
            validate_outputs(@outputs_guessed, @width_indexes.size)

            # close tie beteen step_calc_forward_1 and step_calc_forward_2 as fastest
            @outputs_guessed = @width_indexes.map do |w|
              sum = 0.0
              @height_indexes.each do |h|
                # puts "v"*10
                # puts "@inputs_given: #{@inputs_given}"
                # puts "@weights: #{@weights}"
                # puts "h: #{h}, w: #{w}"
                # puts "-"*10

                sum += @inputs_given[h]*@weights[h][w]
              end
              propagation_function.call(sum)
              # sum
            end

            validate_outputs(@outputs_guessed, @width)
            validate_outputs(@outputs_guessed, @width_indexes.size)
          end
        end
      end
    end
  end
end
