require "./../../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainingData do
  time_col_qty = 3
  config = Ai4cr::NeuralNetwork::Cmn::RnnConcerns::Config.new(
    input_state_size: 11, hidden_state_size: 22, output_state_size: 11,
    time_col_qty: time_col_qty
  )

  simple_wave_rise = (0..10).to_a.map { |i| (0..10).to_a.map { |j| i == j ? 1.0 : 0.0 } }
  training_data = simple_wave_rise + simple_wave_rise.reverse + simple_wave_rise + simple_wave_rise.reverse

  puts "TrainingData.new: #{Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainingData.new}"

  # describe "when importing and exporting as JSON" do
  #   rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new
  #   rnn_to_json = rnn.to_json
  #   rnn_from_json = Ai4cr::NeuralNetwork::Cmn::Rnn.from_json(rnn_to_json)
  #   re_exported_json = rnn_from_json.to_json

  #   # puts rnn_from_json.pretty_inspect

  #   context "when comparing exported vs imported matches for" do
  #     it "config" do
  #       re_exported_json["config"].should eq(rnn_to_json["config"])
  #     end

  #     it "layer_index_max" do
  #       re_exported_json["layer_index_max"].should eq(rnn_to_json["layer_index_max"])
  #     end

  #     it "layer_range" do
  #       re_exported_json["layer_range"].should eq(rnn_to_json["layer_range"])
  #     end

  #     it "time_col_index_max" do
  #       re_exported_json["time_col_index_max"].should eq(rnn_to_json["time_col_index_max"])
  #     end

  #     it "time_col_range" do
  #       re_exported_json["time_col_range"].should eq(rnn_to_json["time_col_range"])
  #     end

  #     it "mini_net_configs" do
  #       re_exported_json["mini_net_configs"].should eq(rnn_to_json["mini_net_configs"])
  #     end

  #     it "mini_net_set" do
  #       re_exported_json["mini_net_set"].should eq(rnn_to_json["mini_net_set"])
  #     end
  #   end
  # end

end
