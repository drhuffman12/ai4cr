module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleManager < Breed::Manager(RnnSimple)
        include JSON::Serializable

        getter mini_net_manager = NeuralNetwork::Cmn::MiniNetManager.new

        class_getter counter : CounterSafe::Exclusive = CounterSafe::Exclusive.new

        def initialize; end

        def breed_validations(parent_a : T, parent_b : T, delta)
          super
          raise Ai4cr::Breed::StructureError.new unless (
            parent_a.history_size == parent_b.history_size &&
            parent_a.io_offset == parent_b.io_offset &&
            parent_a.time_col_qty == parent_b.time_col_qty &&
            parent_a.input_size == parent_b.input_size &&
            parent_a.output_size == parent_b.output_size &&
            parent_a.hidden_layer_qty == parent_b.hidden_layer_qty &&
            parent_a.hidden_size_given == parent_b.hidden_size_given &&
            parent_a.bias_disabled == parent_b.bias_disabled &&
            # parent_a.bias_default == parent_b.bias_default &&
            parent_a.learning_style == parent_b.learning_style
          )
        end

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
