require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::Rnn do
  describe "when importing and exporting as JSON" do
    # TODO
    puts
    puts "v"*20
    puts

    rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new
    puts "rnn.to_json: #{rnn.to_json}"
    puts

    rnn_to_json = rnn.to_json
    rnn_from_json = Ai4cr::NeuralNetwork::Cmn::Rnn.from_json(rnn_to_json)

    puts
    puts "rnn_from_json.to_json == rnn.to_json: #{rnn_from_json.to_json == rnn.to_json}"
    puts
    puts "^"*20
    puts
  end

  # NOTE Below are all for learing style Sigmoid; tests should be added to cover the other learning styles
  describe "#eval" do
    # TODO
  end

  describe "#train" do
    # TODO
  end
end
