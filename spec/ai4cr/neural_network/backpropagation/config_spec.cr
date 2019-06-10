require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Backpropagation::Config do
  describe "#initialize" do
    describe "when given a config with structure of [4, 2] and bias enabled" do
      expected_input_size = 4
      expected_hidden_layer_sizes = [3]
      expected_output_size = 2
      structure = [expected_input_size] + expected_hidden_layer_sizes + [expected_output_size]
      disable_bias = false
      config = Ai4cr::NeuralNetwork::Backpropagation::Config.new(structure, disable_bias) # .init_network

      it "sets @input_size to expected expected_input_size array" do
        config.input_size.should eq(expected_input_size)
      end

      it "sets @hidden_layer_sizes to expected expected_hidden_layer_sizes array" do
        config.hidden_layer_sizes.should eq(expected_hidden_layer_sizes)
      end

      it "sets @expected_output_size to expected expected_output_size array" do
        config.output_size.should eq(expected_output_size)
      end
      
      describe "when exported as json" do
        config_exported = config.to_json
        config_exported_reparsed = JSON.parse(config_exported)

        describe "and re-imported from json" do
          config_reimported = Ai4cr::NeuralNetwork::Backpropagation::Config.from_json(config_exported)
          config_reimported_reparsed = JSON.parse(config_reimported.to_json)

          it "sets @input_size to expected expected_input_size array" do
            config_reimported.input_size.should eq(expected_input_size)
          end
    
          it "sets @hidden_layer_sizes to expected expected_hidden_layer_sizes array" do
            config_reimported.hidden_layer_sizes.should eq(expected_hidden_layer_sizes)
          end
    
          it "sets @expected_output_size to expected expected_output_size array" do
            config_reimported.output_size.should eq(expected_output_size)
          end
    
          it "exported json matches reimported json" do
            config_exported_reparsed.should eq(config_reimported_reparsed)
          end
        end
      end
    end
  end
end
