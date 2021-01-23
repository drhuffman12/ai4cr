require "./../spec_helper"
require "./../spectator_helper"

class Bp
  include Ai4cr::BreedParent
end

Spectator.describe Ai4cr::BreedParent do
  let(breed_parent) { Bp.new }

  describe "#init_name" do
    context "given a string for 'name_suffix'" do
      context "returns" do
        let(fake_utc) { double }
        let(fake_ms) { 1234567890123 }
        let(given_name_suffix) { "abc123" }

        let(result) { breed_parent.init_name(name_suffix: given_name_suffix) }
        let(json_any) { JSON.parse(result) }

        it "a JSON string" do
          expect(result).to be_an(String)
          expect(json_any).to be_an(JSON::Any)
        end

        context "a JSON string with key" do
          it "klass" do
            expect(json_any["klass"]?).not_to be_nil
            expect(json_any["klass"]?).to eq("Bp")
          end

          # before_each do
          #   # TODO: Fix mocking of Time
          #   allow(Time).to receive(utc).and_return(fake_utc)
          #   allow(fake_utc).to receive(to_unix_ms).and_return(fake_ms)
          # end
          it "ms" do
            expect(json_any["ms"]?).not_to be_nil
            # expect(json_any["ms"]?).to eq(fake_ms)
          end

          it "suffix" do
            expect(json_any["suffix"]?).not_to be_nil
            expect(json_any["suffix"]?).to eq(given_name_suffix)
          end
        end
      end
    end
  end

  describe ".breed" do
    let(parent_a) { "Some Object" }
    let(parent_b) { "Other Object" }
    let(delta) { (rand*2 - 0.5) }
    let(name_suffix) { "#{Faker::Name.name}#{rand(1000)}" }

    it "foo" do
      expect {
        breed_parent.breed_with(parent_b)
      }.to raise_error("TO BE IMPLEMENTED IN REAL CLASS")
    end
  end
end
