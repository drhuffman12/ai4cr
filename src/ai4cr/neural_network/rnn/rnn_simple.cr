require "./rnn_simple_concerns/calc_guess.cr"
require "./rnn_simple_concerns/props_and_inits.cr"
require "./rnn_simple_concerns/train_and_adjust.cr"
require "./rnn_simple_concerns/roll_ups.cr"
require "./rnn_simple_concerns/data_utils.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimple
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params

        include JSON::Serializable

        # include Ai4cr::BreedParent(self.class)

        include RnnSimpleConcerns::PropsAndInits
        include RnnSimpleConcerns::CalcGuess
        include RnnSimpleConcerns::TrainAndAdjust
        include RnnSimpleConcerns::RollUps
        include RnnSimpleConcerns::DataUtils

        # re 'Ai4cr::BreedParent'
        def final_li_output_error_distances
          li = synaptic_layer_indexes.last
          time_col_indexes.map do |ti|
            mini_net_set[li][ti].error_stats.distance
          end
        end

        def calculate_error_distance
          @error_stats.distance = final_li_output_error_distances.map { |e| 0.5*(e)**2 }.sum
        end
      end
    end
  end
end
