require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new }

  describe "#to_json" do
    context "correctly exports" do
      it "the whole initial object" do
        counter = my_breed_manager.counter
        puts_debug
        puts_debug "counter: #{counter.to_json}"
        puts_debug

        exported_json = my_breed_manager.to_json
        expected_json = "{\"mini_net_manager\":{}}"

        expect(exported_json).to be_a(String)
        expect(exported_json).to eq(expected_json)
      end
    end
  end

  describe "#to_json and #from_json" do
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
