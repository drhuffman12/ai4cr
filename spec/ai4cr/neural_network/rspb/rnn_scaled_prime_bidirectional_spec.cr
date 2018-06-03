require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional do
  described_class = Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional

  describe "initializer methods" do
    describe "#initialize" do
      describe "when passed no params" do
        subject = described_class.new

        describe "uses default values for" do
          it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PRIME_QTY_DEFAULT}" do
            subject.prime_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PRIME_QTY_DEFAULT)
          end

          it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::ZOOM_QTY_DEFAULT}" do
            subject.zoom_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::ZOOM_QTY_DEFAULT)
          end

          it "prime_qty of #{Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PANEL_QTY_DEFAULT}" do
            subject.panel_qty.should eq(Ai4cr::NeuralNetwork::Rspb::RnnScaledPrimeBidirectional::PANEL_QTY_DEFAULT)
          end
        end

        it "determines correct value for io_min_qty_required" do
          io_min_qty_required_expected = subject.zoomed_prime_offsets.last.last * 2 + 1
          subject.io_min_qty_required.should eq(io_min_qty_required_expected)
        end
      end
    end
  end
end
