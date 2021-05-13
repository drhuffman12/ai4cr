module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module Common
          module TrainAndAdjust
            getter output_set_expected = Array(Array(Float64)).new
            getter all_output_errors = Array(Array(Float64)).new

            def calculate_error_distance
              @error_stats.distance = final_li_output_error_distances.sum { |e| 0.5*(e)**2 }
            end
          end
        end
      end
    end
  end
end
