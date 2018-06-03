require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rspb::Util do
  described_class = Ai4cr::NeuralNetwork::Rspb::Util

  describe "class methods" do
    describe ".gen_prime_offsets" do
      describe "returns an array of prime-like positive and negative offsets (with -1, 0, and 1 included)" do
        it "when given prime_qty of 3" do
          prime_qty = 3
          prime_offsets_expected = [-3, -2, -1, 0, 1, 2, 3]
          described_class.gen_prime_offsets(prime_qty).should eq(prime_offsets_expected)
        end

        it "when given prime_qty of 8" do
          prime_qty = 8
          prime_offsets_expected = [-34, -21, -13, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 13, 21, 34]
          described_class.gen_prime_offsets(prime_qty).should eq(prime_offsets_expected)
        end
      end
    end
    
    describe ".gen_zoom_scales" do
      describe "returns an array of base-2 exponential values" do
        it "when given zoom_qty of 3" do
          zoom_qty = 3
          zoom_scales_expected = [1, 2, 4]
          described_class.gen_zoom_scales(zoom_qty).should eq(zoom_scales_expected)
        end

        it "when given zoom_qty of 8" do
          zoom_qty = 8
          zoom_scales_expected = [1, 2, 4, 8, 16, 32, 64, 128]
          described_class.gen_zoom_scales(zoom_qty).should eq(zoom_scales_expected)
        end
      end
    end
    
    describe ".gen_zoomed_prime_offsets" do
      describe "returns an 2-dim array with inner array based on gen_prime_offsets but each successive inner array zoomed out" do
        it "when given zoom_qty of 3 and prime_qty of 3" do
          zoom_qty = 3
          prime_qty = 3
          zoomed_prime_expected = [[-3, -2, -1, 0, 1, 2, 3], [-6, -4, -2, 0, 2, 4, 6], [-12, -8, -4, 0, 4, 8, 12]]
          described_class.gen_zoomed_prime_offsets(zoom_qty, prime_qty).should eq(zoomed_prime_expected)
        end

        it "when given zoom_qty of 8 and prime_qty of 8" do
          zoom_qty = 8
          prime_qty = 8
          zoomed_prime_expected = [
            [-34, -21, -13, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 13, 21, 34],
            [-68, -42, -26, -16, -10, -6, -4, -2, 0, 2, 4, 6, 10, 16, 26, 42, 68],
            [-136, -84, -52, -32, -20, -12, -8, -4, 0, 4, 8, 12, 20, 32, 52, 84, 136],
            [-272, -168, -104, -64, -40, -24, -16, -8, 0, 8, 16, 24, 40, 64, 104, 168, 272],
            [-544, -336, -208, -128, -80, -48, -32, -16, 0, 16, 32, 48, 80, 128, 208, 336, 544],
            [-1088, -672, -416, -256, -160, -96, -64, -32, 0, 32, 64, 96, 160, 256, 416, 672, 1088],
            [-2176, -1344, -832, -512, -320, -192, -128, -64, 0, 64, 128, 192, 320, 512, 832, 1344, 2176], [-4352, -2688, -1664, -1024, -640, -384, -256, -128, 0, 128, 256, 384, 640, 1024, 1664, 2688, 4352]
          ]
          described_class.gen_zoomed_prime_offsets(zoom_qty, prime_qty).should eq(zoomed_prime_expected)
        end
      end
    end
  end
end
