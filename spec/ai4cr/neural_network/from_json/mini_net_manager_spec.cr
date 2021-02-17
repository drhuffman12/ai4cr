require "./../../../spec_helper"
require "./../../../spectator_helper"

def puts_debug(message = "")
  puts message if ENV.has_key?("DEBUG") && ENV["DEBUG"] == "1"
end

Spectator.describe Ai4cr::NeuralNetwork::Cmn::MiniNetManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }

  before_each do
    my_breed_manager.counter.reset!
  end

  describe "#to_json" do
    context "for a new MiniNetManager" do
      context "correctly exports" do
        it "the whole initial object" do
          # my_breed_manager.counter.inc("foo")
          counter = my_breed_manager.counter
          puts_debug
          puts_debug "counter.to_json: #{counter.to_json}"
          puts_debug

          # NOTE: 'exported' vs 'expected'
          exported_json = my_breed_manager.to_json
          expected_json = "{}"

          expect(exported_json).to be_a(String)
          expect(exported_json).to eq(expected_json)
        end
      end
    end
  end

  describe "#to_json and #from_json" do
    context "for a new MiniNetManager" do
      context "correctly exports and imports" do
        it "the whole object" do
          exported_json = my_breed_manager.to_json
          imported = my_breed_manager.class.from_json(exported_json)
          re_exported_json = imported.to_json

          expect(re_exported_json).to eq(exported_json)
        end
      end
    end
  end
end
