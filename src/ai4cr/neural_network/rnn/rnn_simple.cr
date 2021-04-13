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

        include Ai4cr::Breed::Client
        include RnnSimpleConcerns::PropsAndInits
        include RnnSimpleConcerns::CalcGuess
        include RnnSimpleConcerns::TrainAndAdjust
        include RnnSimpleConcerns::RollUps
        include RnnSimpleConcerns::DataUtils

        def clone
          a_clone = RnnSimple.new(
            name: self.name.clone,

            history_size: self.history_size.clone,

            io_offset: self.io_offset.clone,
            time_col_qty: self.time_col_qty.clone,
            input_size: self.input_size.clone,
            output_size: self.output_size.clone,
            hidden_layer_qty: self.hidden_layer_qty.clone,
            hidden_size_given: self.hidden_size_given.clone,

            learning_styles: self.learning_styles.clone,

            bias_disabled: self.bias_disabled.clone,
            bias_default: self.bias_default.clone,

            learning_rate: self.learning_rate.clone,
            momentum: self.momentum.clone,
            deriv_scale: self.deriv_scale.clone,
          )
          a_clone.mini_net_set = self.mini_net_set.clone

          a_clone
        end
      end
    end
  end
end
