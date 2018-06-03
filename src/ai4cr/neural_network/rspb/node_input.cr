require "./concerns/outs"

# macro alias_method(old_method, new_method)
#   def {{new_method.id}}(*args)
#     {{old_method.id}}(*args)
#   end
# end

# Ai4cr::NeuralNetwork::Rspb::NodeInput
module Ai4cr
  module NeuralNetwork
    module Rspb
      # struct PrevMem
      class NodeInput
        include Concerns::Outs # memory

        def initialize(@outputs_size)
          init_outputs
        end

        # AKA: alias_method :init_outputs, :load
        def load(outputs_new)
          init_outputs(outputs_new)
        end
      end
    end
  end
end

