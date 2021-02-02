module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleManager < Breed::Manager(RnnSimple)

        def copy_and_mix(parent_a, parent_b, delta)
          # TODO (probably ok)
          child = parts_to_copy(parent_a, parent_b, delta)
          mix_parts(child, parent_a, parent_b, delta)
        end

        def parts_to_copy(parent_a : RnnSimple, parent_b : RnnSimple, delta)
          # TODO (probably ok)
          T.from_json(parent_a.to_json)
        end
        
        def mix_parts(child : RnnSimple, parent_a : RnnSimple, parent_b : RnnSimple, delta)
          # TODO (probably: for each li and for each ti, breed associated MiniNet)
          child
        end
      end
    end
  end
end
