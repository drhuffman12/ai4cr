require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rspb::NodeInput do
  described_class = Ai4cr::NeuralNetwork::Rspb::NodeInput

  describe "initializer methods" do
    describe "#initialize" do
      describe "when passed an outputs_size" do
        it "sets @outputs_size to the given outputs_size" do
          outputs_size_given = 4

          ni = described_class.new(outputs_size_given)

          ni.outputs_size.should eq(outputs_size_given)
        end

        it "sets @outputs to an array of the given outputs_size where all elements are zero" do
          outputs_size_given = 4
          # outputs_default = [0.0]
          outputs_expected = [0.0, 0.0, 0.0, 0.0]

          ni = described_class.new(outputs_size_given)

          # ni.outputs.should eq(outputs_default)
          ni.outputs.should eq(outputs_expected)
        end
      end
    end
  end

  describe "instance methods" do

    # AKA: alias_method :init_outputs, :load
    describe "#load" do
      describe "when passed an outputs_new" do
        outputs_size_given = 4
        ni = described_class.new(outputs_size_given)

        describe "of incorrect size" do
          it "raises an ArgumentError" do
            expect_raises(ArgumentError) do
              outputs_new = [0.0, 1.0]
    
              ni.load(outputs_new)
            end
          end
        end

        describe "of correct size" do
          it "sets @outputs to the given outputs_new" do
            outputs_new = [0.0, 0.1, 0.2, 0.3]
  
            ni.load(outputs_new)
  
            ni.outputs.should eq(outputs_new)
          end

          it "returns self" do
            outputs_new = [0.0, 0.1, 0.2, 0.3]
  
            ni.load(outputs_new).should eq(ni)
          end
        end
      end
    end

  end
end


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
