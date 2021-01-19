require "./../spec_helper"
require "./../spectator_helper"

Spectator.describe Ai4cr::BreedUtils do
  let(parents1) { (-10..0).to_a.map { |i| i/10.0 } }
  let(parents2) { (0..10).to_a.map { |i| i/10.0 } }

  let(breeder) {
    Ai4cr::Breeder.new
  }

  describe "breed_value" do
    let(parent_a) { 0.0 }
    let(parent_b) { 1.0 }
    let(p_dist) { 1.0 * (parent_b - parent_a) }
    let(c_min) { parent_a - 0.5 }
    let(c_max) { parent_a + 2 * p_dist - 0.5 }

    context "debug" do
      it "p_dist" do
        expect(p_dist).to eq(1.0)
      end

      it "c_min" do
        expect(c_min).to eq(-0.5)
      end

      it "c_max" do
        expect(c_max).to eq(1.5)
      end
    end

    context "returns" do
      it "a Float64" do
        child = breeder.breed_value(0, 1)

        expect(child).to be_a(Float64)
      end

      context "between" do
        it "expected 'min'" do
          child = breeder.breed_value(0, 1)

          expect(child).to be >= c_min
        end

        it "expected 'max'" do
          child = breeder.breed_value(0, 1)

          expect(c_max).to be >= child
        end
      end
    end
  end

  describe "breed_nested" do
    context "given two Int32 values" do
      let(parent_a) { 0 }
      let(parent_b) { 1 }

      it "does not raise" do
        expect {
          breeder.breed_nested(parent_a, parent_b)
        }.not_to raise_error
      end

      context "returns" do
        it "a Float64" do
          child = breeder.breed_nested(parent_a, parent_b)

          expect(child).to be_a(Float64)
        end
      end
    end

    context "given two Float64 values" do
      let(parent_a) { 0.0 }
      let(parent_b) { 1.0 }

      it "does not raise" do
        expect {
          breeder.breed_nested(parent_a, parent_b)
        }.not_to raise_error
      end

      context "returns" do
        it "a Float64" do
          child = breeder.breed_nested(parent_a, parent_b)

          expect(child).to be_a(Float64)
        end
      end
    end

    context "given two Array(Float64) values" do
      let(parent_a) { [0.0, 0.1] }
      let(parent_b) { [0.9, 1.0] }

      it "does not raise" do
        expect {
          breeder.breed_nested(parent_a, parent_b)
        }.not_to raise_error
      end

      context "returns" do
        it "a Float64" do
          child = breeder.breed_nested(parent_a, parent_b)

          expect(child).to be_a(Array(Float64))
        end
      end
    end

    # context "given two Hash(String, Float64) values" do
    #   let(parent_a) {
    #     Hash{"zero" => 0.0}
    #   }
    #   let(parent_b) {
    #     Hash{"one" => 1.0}
    #   }

    #   it "does not raise" do
    #     expect {
    #       breeder.breed_nested(parent_a, parent_b)
    #     }.not_to raise_error
    #   end

    #   context "returns" do
    #     it "a Float64" do
    #       child = breeder.breed_nested(parent_a, parent_b)

    #       expect(child).to be_a(parent_a.class)
    #     end
    #   end
    # end
  end
end
