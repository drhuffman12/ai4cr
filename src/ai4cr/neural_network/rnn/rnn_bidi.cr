require "./rnn_simple_concerns/calc_guess.cr"
require "./rnn_simple_concerns/props_and_inits.cr"
require "./rnn_simple_concerns/train_and_adjust.cr"
require "./rnn_simple_concerns/roll_ups.cr"

require "./rnn_bi_di_concerns/calc_guess.cr"
require "./rnn_bi_di_concerns/props_and_inits.cr"
require "./rnn_bi_di_concerns/train_and_adjust.cr"
require "./rnn_bi_di_concerns/roll_ups.cr"

require "./rnn_simple_concerns/data_utils.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnBiDi
        # TODO: Implement Bi-directional RNN (i.e.: RnnSimple pulls from inputs and previous time column.)
        # This class must also pull from next time column and mix them all together in subsequent hidden layers.

        include JSON::Serializable

        include Ai4cr::Breed::Client

        include RnnSimpleConcerns::PropsAndInits
        include RnnBiDiConcerns::PropsAndInits

        include RnnSimpleConcerns::CalcGuess
        include RnnBiDiConcerns::CalcGuess

        include RnnSimpleConcerns::TrainAndAdjust
        include RnnBiDiConcerns::TrainAndAdjust

        include RnnSimpleConcerns::RollUps
        include RnnBiDiConcerns::RollUps

        include RnnSimpleConcerns::DataUtils

        def clone
          a_clone = RnnBiDi.new(
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
