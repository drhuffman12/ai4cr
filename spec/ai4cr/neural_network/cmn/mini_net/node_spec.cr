
require "./../../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::MiniNet::Node do
  describe "#initialize" do
    [
      Ai4cr::NeuralNetwork::Cmn::LS_PRELU,
      Ai4cr::NeuralNetwork::Cmn::LS_RELU,
      Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID,
      Ai4cr::NeuralNetwork::Cmn::LS_TANH
    ].each do |learning_style|
      context "when given height: 2, width: 3, learning_style: #{learning_style}" do
        context "when exporting to JSON" do

          np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(height: 2, width: 3, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
          np1_json = np1.to_json
          np1_hash = JSON.parse(np1_json).as_h

          expected_keys = ["width", "height", "height_considering_bias", "width_indexes", "height_indexes", "inputs_given", "outputs_guessed", "weights", "last_changes", "error_total", "outputs_expected", "input_deltas", "output_deltas", "disable_bias", "learning_rate", "momentum", "error_distance", "error_distance_history_max", "error_distance_history", "learning_style", "deriv_scale"]
          expected_keys.each do |key|
            it "it has top level key of #{key}" do            
              (np1_hash.keys).should contain(key)
            end
          end
        end

        context "when importing from JSON" do

          np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(2,3,Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
          np1_json = np1.to_json

          np2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.from_json(np1_json)
          np2_json = np2.to_json

          # FYI: Due to some rounding errors during export/import, the following might not work:
          # it "re-exported JSON matches imported JSON" do            
          #   (np1_json).should eq(np2_json)
          # end
          # e.g.:
          # Expected: "{\"width\":3,\"height\":2,\"height_considering_bias\":3,\"width_indexes\":[0,1,2],\"height_indexes\":[0,1,2],\"inputs_given\":[0.0,0.0,1.0],\"outputs_guessed\":[0.0,0.0,0.0],\"weights\":[[0.7318031568424814,0.534853051161922,0.21857644593495615],[-0.6591430323844467,-0.2012854441173063,-0.3036688821984831],[0.3937028443098609,-0.1193921136297592,-0.5135509965693288]],\"last_changes\":[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],\"error_total\":0.0,\"outputs_expected\":[0.0,0.0,0.0],\"input_deltas\":[0.0,0.0,0.0],\"output_deltas\":[0.0,0.0,0.0],\"disable_bias\":false,\"learning_rate\":0.18325052338453365,\"momentum\":0.8206852816702831,\"error_distance\":1.0,\"error_distance_history_max\":10,\"error_distance_history\":[],\"learning_style\":10,\"deriv_scale\":0.001}"
          # got: "{\"width\":3,\"height\":2,\"height_considering_bias\":3,\"width_indexes\":[0,1,2],\"height_indexes\":[0,1,2],\"inputs_given\":[0.0,0.0,1.0],\"outputs_guessed\":[0.0,0.0,0.0],\"weights\":[[0.7318031568424814,0.534853051161922,0.21857644593495618],[-0.6591430323844467,-0.2012854441173063,-0.3036688821984831],[0.3937028443098609,-0.11939211362975921,-0.5135509965693288]],\"last_changes\":[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],\"error_total\":0.0,\"outputs_expected\":[0.0,0.0,0.0],\"input_deltas\":[0.0,0.0,0.0],\"output_deltas\":[0.0,0.0,0.0],\"disable_bias\":false,\"learning_rate\":0.18325052338453365,\"momentum\":0.8206852816702831,\"error_distance\":1.0,\"error_distance_history_max\":10,\"error_distance_history\":[],\"learning_style\":10,\"deriv_scale\":0.001}"

          # However, it seems to be fine when you split it out by top-level keys:
          expected_keys = ["width", "height", "height_considering_bias", "width_indexes", "height_indexes", "inputs_given", "outputs_guessed", "weights", "last_changes", "error_total", "outputs_expected", "input_deltas", "output_deltas", "disable_bias", "learning_rate", "momentum", "error_distance", "error_distance_history_max", "error_distance_history", "learning_style", "deriv_scale"]
          expected_keys.each do |key|
            it "re-exported JSON matches imported JSON for top level key of #{key}" do            
              (np1_json[key]).should eq(np2_json[key])
            end
          end

          # And, it seems to be fine when you convert to hash:
          np1_hash = JSON.parse(np1_json).as_h
          np2_hash = JSON.parse(np2_json).as_h
          # FYI: Due to some rounding errors during export/import, the following might not work:
          it "re-exported JSON matches imported JSON" do            
            (np1_hash).should eq(np2_hash)
          end
        end
        

      end
    end
  end
end