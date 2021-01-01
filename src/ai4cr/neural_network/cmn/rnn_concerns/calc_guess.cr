require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module CalcGuess

          def outputs_guessed
            li = synaptic_layer_indexes.last

            mn_output_size = node_output_sizes[li]
            time_col_indexes.map do |ti|
              mini_net_set[li][ti].outputs_guessed
            end
          end
          
          # steps for 'eval' aka 'guess':
          def eval(input_set_given)
            step_load_inputs(input_set_given)
            step_calc_forward

            # outputs_guessed
            [[-10.0], [-20.0]] 
          end

          def step_load_inputs(input_set_given)
            # TODO
          end

          def step_calc_forward
            # TODO
          end

        end
      end
    end
  end
end
