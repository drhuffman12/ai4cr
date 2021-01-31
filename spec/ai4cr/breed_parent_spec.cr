require "./../spec_helper"
require "./../spectator_helper"

class Bp1
  getter name : String

  include JSON::Serializable
  include Ai4cr::BreedParent

  def initialize(name_suffix = "one") # Time.utc.to_s
    @name = init_name(name_suffix)
  end
end
class Bp2
  getter name : String

  include JSON::Serializable
  include Ai4cr::BreedParent

  def initialize(name_suffix = "one") # Time.utc.to_s
    @name = init_name(name_suffix)
  end
end

Spectator.describe Ai4cr::BreedParent do
  let(breed_parent) { Bp1.new }

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
            expect(json_any["klass"]?).to eq("Bp1")
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

  describe ".breed_with" do
    let(delta) { (rand*2 - 0.5) }
    let(name_suffix) { "#{Faker::Name.name}#{rand(1000)}" }

    let(parent_a) { Bp1.new }
    let(parent_b) { Bp2.new }
    let(parent_c) { Bp1.new }

    context "an instance of a MIS-matching class" do
      it "RAISES a Ai4cr::BreedMismatch exception" do
        error_message_expected = "Breed Class Mis-match; parent_a.class.name: #{parent_a.class.name},  parent_b.class.name: #{parent_b.class.name}"
  
        expect {
          parent_a.breed_with(parent_b)
        }.to raise_error(Ai4cr::BreedMismatch, error_message_expected)
      end
    end

    context "an instance of a MATCHING class" do
      it "does NOT raise a Ai4cr::BreedMismatch exception" do
        expect {
          parent_a.breed_with(parent_c)
        }.not_to raise_error
      end

      context "returns" do
        it "a child of a matching class" do
          child = parent_a.breed_with(parent_c)
          expect(child).to be_a(Bp1)
          expect(child).to be_a(Bp1)
        end
      end
    end
  end
end
