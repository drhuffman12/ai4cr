require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::Rnn do
  describe "when importing and exporting as JSON" do
    rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new
    rnn_to_json = rnn.to_json
    rnn_from_json = Ai4cr::NeuralNetwork::Cmn::Rnn.from_json(rnn_to_json)
    re_exported_json = rnn_from_json.to_json

    puts rnn_from_json.pretty_inspect

    context "when comparing exported vs imported matches for" do
      it "config" do
        re_exported_json["config"].should eq(rnn_to_json["config"])
      end

      it "layer_index_max" do
        re_exported_json["layer_index_max"].should eq(rnn_to_json["layer_index_max"])
      end

      it "layer_range" do
        re_exported_json["layer_range"].should eq(rnn_to_json["layer_range"])
      end

      it "time_col_index_max" do
        re_exported_json["time_col_index_max"].should eq(rnn_to_json["time_col_index_max"])
      end

      it "time_col_range" do
        re_exported_json["time_col_range"].should eq(rnn_to_json["time_col_range"])
      end

      it "mini_net_configs" do
        re_exported_json["mini_net_configs"].should eq(rnn_to_json["mini_net_configs"])
      end

      it "mini_net_set" do
        re_exported_json["mini_net_set"].should eq(rnn_to_json["mini_net_set"])
      end
    end
  end

  describe "#eval" do
    # TODO
    time_col_qty = 3
    config = Ai4cr::NeuralNetwork::Cmn::RnnConcerns::NetConfig.new(
      input_state_size: 11, hidden_state_size: 22, output_state_size: 11,
      time_col_qty: time_col_qty      
    )
    rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)

    puts "\nBEFORE:\n"
    puts rnn.to_json.pretty_inspect

    simple_wave_rise = (0..10).to_a.map{ |i| (0..10).to_a.map{ |j| i == j ? 1.0 : 0.0 } }
    training_data = simple_wave_rise + simple_wave_rise.reverse + simple_wave_rise + simple_wave_rise.reverse

    # eval
    offset = 0
    time_col_from = offset
    time_col_to = offset + time_col_qty - 1
    rnn.eval(training_data[time_col_from..time_col_to])
    
    puts "\nAFTER:\n"
    puts rnn.to_json.pretty_inspect
    puts "\nrnn.outputs_guessed:\n"
    puts rnn.outputs_guessed
    puts "\nrnn.guesses_best:\n"
    puts rnn.guesses_best
    # # train
    # training_data_size = training_data.size
    # time_col_qty
    
  end

  describe "#train" do
    # TODO
  end
end
