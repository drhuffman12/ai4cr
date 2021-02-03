module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleManager < Breed::Manager(RnnSimple)
        getter mini_net_manager = Cmn::MiniNetManager.new

        def mix_parts(child : RnnSimple, parent_a : RnnSimple, parent_b : RnnSimple, delta)
          bias_default = mix_one_part_number(parent_a.bias_default, parent_b.bias_default, delta)
          child.bias_default = bias_default

          learning_rate = mix_one_part_number(parent_a.learning_rate, parent_b.learning_rate, delta)
          child.learning_rate = learning_rate

          momentum = mix_one_part_number(parent_a.momentum, parent_b.momentum, delta)
          child.momentum = momentum

          deriv_scale = mix_one_part_number(parent_a.deriv_scale, parent_b.deriv_scale, delta)
          child.deriv_scale = deriv_scale

          child.synaptic_layer_indexes.each do |li|
            child.time_col_indexes.each do |ti|
              mini_net_a = parent_a.mini_net_set[li][ti]
              mini_net_b = parent_b.mini_net_set[li][ti]

              child.mini_net_set[li][ti] = mini_net_manager.breed(mini_net_a, mini_net_b, delta)
            end
          end

          child
        end
      end
    end
  end
end
