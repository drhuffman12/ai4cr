require "./../../../../spec_helper"

class MockClassIncludingOuts
  include Ai4cr::NeuralNetwork::Rspb::Concerns::Outs
end

describe Ai4cr::NeuralNetwork::Rspb::Concerns::Outs do
  # described_module = Ai4cr::NeuralNetwork::Rspb::Concerns::Outs

  described_class = MockClassIncludingOuts

  describe "instance variables are set to defaults" do
    it ".outputs_size" do
      outputs_size_expected = 0

      outs = described_class.new

      outs.outputs_size.should eq(outputs_size_expected)
    end

    it ".outputs_size" do
      outputs_expected = [0.0]

      outs = described_class.new

      outs.outputs.should eq(outputs_expected)
    end
  end

  describe "instance methods" do
    describe "#init_outputs" do
      describe "when given nil" do
        outs = described_class.new
        outputs_new = nil
        
        # it "leaves the @outputs_size to the default" do
        #   outputs_new = nil
        #   outputs_size_expected = outputs_new.size

        #   outs = described_class.new
        #   outs.init_outputs(outputs_new)

        #   outs.outputs_size.should eq(outputs_size_expected)
        # end

        it "resets the @outputs to an array of zeros where the array size is outputs_size" do
          outputs_expected = (1..outs.outputs_size).map { 0.0 }

          outs.init_outputs(outputs_new)

          outs.outputs.should eq(outputs_expected)
        end

        it "returns self" do
          outs.init_outputs(outputs_new).should eq(outs)
        end
      end

      describe "when passed an outputs_new" do
        outputs_size_given = 4
        ni = described_class.new
        ni.outputs_size = outputs_size_given

        describe "of incorrect size" do
          it "raises an ArgumentError" do
            expect_raises(ArgumentError) do
              outputs_new = [0.0, 1.0]
    
              ni.init_outputs(outputs_new)
            end
          end
        end

        describe "of correct size" do
          it "sets @outputs to the given outputs_new" do
            outputs_new = [0.0, 0.1, 0.2, 0.3]
  
            ni.init_outputs(outputs_new)
  
            ni.outputs.should eq(outputs_new)
          end

          it "returns self" do
            outputs_new = [0.0, 0.1, 0.2, 0.3]
  
            ni.init_outputs(outputs_new).should eq(ni)
          end
        end
      end

      describe "when given an array of two floats" do
        describe "of incorrect size" do
          it "raises an ArgumentError" do
            expect_raises(ArgumentError) do
              outputs_new = [0.1, 0.2]
              
              outs = described_class.new

              outs.init_outputs(outputs_new)
            end
          end
        end

        # it "sets the @outputs_size to 2" do
        #   outputs_new = [0.1, 0.2]
        #   outputs_size_expected = outputs_new.size

        #   outs = described_class.new
        #   outs.init_outputs(outputs_new)

        #   outs.outputs_size.should eq(outputs_size_expected)
        # end

        # it "sets the @outputs to match the given outputs_new" do
        #   outputs_new = [0.1, 0.2]

        #   outs = described_class.new
        #   outs.init_outputs(outputs_new)

        #   outs.outputs.should eq(outputs_new)
        # end

        # it "returns self" do
        #   outputs_new = [0.1, 0.2]

        #   outs = described_class.new
        #   outs.init_outputs(outputs_new).should eq(outs)
        # end
      end
    end
  end

end