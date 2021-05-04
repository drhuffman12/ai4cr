require "./concerns/common/props_and_inits.cr"
require "./concerns/bi_di/pai_distinct.cr"
require "./concerns/bi_di/aliases.cr"
require "./concerns/bi_di/calc_guess.cr"
require "./concerns/common/train_and_adjust.cr"
require "./concerns/common/roll_ups.cr"
require "./concerns/common/data_utils.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnBiDi
        # TODO: Implement Bi-directional RNN (i.e.: RnnSimple pulls from inputs and previous time column.)
        # This class must also pull from next time column and mix them all together in subsequent hidden layers.

        include JSON::Serializable

        include Ai4cr::Breed::Client

        include Concerns::Common::PropsAndInits
        include Concerns::BiDi::PaiDistinct

        # include Concerns::Common::CalcGuess
        include Concerns::BiDi::CalcGuess

        include Concerns::Common::TrainAndAdjust
        # include Concerns::BiDi::TrainAndAdjust

        include Concerns::Common::RollUps
        # include Concerns::BiDi::RollUps

        include Concerns::Common::DataUtils

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
