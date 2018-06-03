require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rspb::NodeMem do
  described_class = Ai4cr::NeuralNetwork::Rspb::NodeMem

  describe "initializer methods" do
    describe "#initialize" do
      describe "when passed an input_node" do
        outputs_size_given = 4
        input_node_given = Ai4cr::NeuralNetwork::Rspb::NodeInput.new(outputs_size_given)
        nm = described_class.new(input_node_given)

        it "sets @input_node to the given input_node" do
          nm.input_node.should eq(input_node_given)
        end

        it "sets @outputs_size to the given input_node's outputs_size" do
          nm.outputs_size.should eq(input_node_given.outputs_size)
        end

        it "leaves @outputs as the default outputs" do
          outputs_default = [0.0]

          nm.outputs.should eq(outputs_default)
        end
      end
    end
  end

  describe "instance methods" do
    describe "#update" do
      it "sets @outputs to a clone of @input_node.outputs" do
        outputs_size_given = 4
        input_node_given = Ai4cr::NeuralNetwork::Rspb::NodeInput.new(outputs_size_given)
        nm = described_class.new(input_node_given)

        outputs_default = [0.0]

        nm.outputs.should eq(outputs_default)

        nm.update

        nm.outputs.should eq(input_node_given.outputs)
      end
    end
  end
end



# ni = Ai4cr::NeuralNetwork::Rspb::NodeInput.new(4)
# inputs = [0.1, 0.2, 0.3, 0.4]
# ni.load(inputs)


      # describe "when passed no params" do
      #   # it "raises" do
      #   #   expect_raises(ArgumentError) do
      #   #     subject = described_class.new
      #   #   end
      #   # end
        

      #   # describe "uses default values for" do
      #   #   it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PRIME_QTY_DEFAULT}" do
      #   #     subject.prime_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PRIME_QTY_DEFAULT)
      #   #   end

      #   #   it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::ZOOM_QTY_DEFAULT}" do
      #   #     subject.zoom_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::ZOOM_QTY_DEFAULT)
      #   #   end

      #   #   it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PANEL_QTY_DEFAULT}" do
      #   #     subject.panel_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PANEL_QTY_DEFAULT)
      #   #   end
      #   # end

      #   # it "determines correct value for io_min_qty_required" do
      #   #   io_min_qty_required_expected = subject.zoomed_prime_offsets.last.last * 2 + 1
      #   #   subject.io_min_qty_required.should eq(io_min_qty_required_expected)
      #   # end
      # end
