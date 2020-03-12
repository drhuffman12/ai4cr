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
  end

  describe "#train" do
    # TODO
  end
end
