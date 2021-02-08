require "./../../../spec_helper"
require "./../../../spectator_helper"

def puts_debug(message = "")
  puts message if ENV.has_key?("DEBUG") && ENV["DEBUG"] == "1"
end

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new }

  # before_each do
  #   my_breed_manager.counter.reset!
  # end

  # describe "#to_json" do
  #   context "correctly exports" do
  #     it "the whole initial object" do
  #       my_breed_manager.counter.inc("foo")
  #       counter = my_breed_manager.counter
  #       puts
  #       puts "counter: #{counter.to_json}"
  #       puts

  #       exported_json = my_breed_manager.to_json
  #       expected_hash = {
  #         "counter" => CounterSafe::AbstractCounter::InternalCounterClass.new(0),
  #         "mini_net_manager" => {
  #           "counter" => CounterSafe::AbstractCounter::InternalCounterClass.new(0)
  #         }
  #       }
  #       expected_json = expected_hash.to_json
  #       expect(exported_json).to be_a(String)
  #       expect(exported_json).to eq(expected_json)
  #     end
  #   end
  # end

  # describe "#to_json and #from_json" do
  #   context "correctly exports and imports" do
  #     it "the whole object" do
  #       exported_json = my_breed_manager.to_json
  #       imported = my_breed_manager.class.from_json(exported)
  #       re_exported_json = imported.to_json

  #       expect(re_exported_json).to eq(exported_json)
  #     end
  #   end
  # end

  # describe "#reset_all" do
  #   context "correctly exports" do
  #     pending "the hash values" do
  #       exported_json = my_breed_manager.values
  #       # imported = my_breed_manager.class.from_json(exported)
  #       # re_exported_json = imported.to_json

  #       my_breed_manager.reset_all(exported_json)

  #       re_exported_json = my_breed_manager.values

  #       expect(re_exported_json).to eq(exported_json)
  #       # We can't to_/from_json when a Mutex is contained (i.e.: in Counter::Safe)
  #       # Currently, we get: Error: no overload matches 'Counter::Safe#to_json' with type JSON::Builder
  #     end
  #   end
  # end
end
