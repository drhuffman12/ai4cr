require "./concerns/outs"

# Ai4cr::NeuralNetwork::Rspb::NodeMem
module Ai4cr
  module NeuralNetwork
    module Rspb
      class NodeMem
        include Concerns::Outs

        getter input_node
        # private def input_node
        #   @input_node
        # end

        def initialize(input_node : Concerns::Outs)
          @input_node = input_node
          @outputs_size = @input_node.outputs_size # outputs.size
        end

        def update
          init_outputs(@input_node.outputs.clone)
          return self
        end
      end
    end
  end
end


          # i = -1
          # inputs_set.each do |inputs|
          #   inputs.each do |in_val|
          #     i += 1
          #     @outputs[i] = in_val
          #   end
          # end


# ni = Ai4cr::NeuralNetwork::Rspb::NodeInput.new(4)
# inputs = [0.1, 0.2, 0.3, 0.4]
# ni.load(inputs)

# nm = Ai4cr::NeuralNetwork::Rspb::NodeMem.new(ni)
# nm.input_node
# nm.update
# nm.input_node