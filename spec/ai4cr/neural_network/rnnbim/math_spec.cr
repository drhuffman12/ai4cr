require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnnbim::Math do
  subject = Ai4cr::NeuralNetwork::Rnnbim::Math

  describe ".node_delta_scale" do
    describe "returns expected scale for a given" do
      it "hiddle_layer_index 0" do
        given = 0
        expected = 1
        subject.node_delta_scale(given).should eq(expected)
      end

      it "hiddle_layer_index 1" do
        given = 1
        expected = 2
        subject.node_delta_scale(given).should eq(expected)
      end

      it "hiddle_layer_index 2" do
        given = 2
        expected = 4
        subject.node_delta_scale(given).should eq(expected)
      end
    end
  end

  describe ".node_scaled_border_past" do
    describe "returns expected scale for a given" do
      it "hiddle_layer_index 0" do
        given = 0
        expected = 1
        subject.node_scaled_border_past(given).should eq(expected)
      end

      it "hiddle_layer_index 1" do
        given = 1
        expected = 2
        subject.node_scaled_border_past(given).should eq(expected)
      end

      it "hiddle_layer_index 2" do
        given = 2
        expected = 4
        subject.node_scaled_border_past(given).should eq(expected)
      end
    end
  end

  describe ".node_scaled_border_future" do
    describe "for a given a" do
      describe "time_column_range (0..3)" do # 
        time_column_range = (0..7)
        describe "returns expected scale for a given" do
          it "hiddle_layer_index 0" do
            given = 0
            expected = 6
            subject.node_scaled_border_future(time_column_range, given).should eq(expected)
          end
    
          it "hiddle_layer_index 1" do
            given = 1
            expected = 5
            subject.node_scaled_border_future(time_column_range, given).should eq(expected)
          end
    
          it "hiddle_layer_index 2" do
            given = 2
            expected = 3
            subject.node_scaled_border_future(time_column_range, given).should eq(expected)
          end
        end
      end
    end
  end

  describe ".simple_weights_sum" do
    it "returns expected values" do
      ins = [0.1,0.2,0.3]
      simple_weights = [[-1.0, 1.0], [0.5, -0.5], [-0.25, 0.25]]
      expected_outs = [-0.075, 0.075]
  
      subject.simple_weights_sum(ins, simple_weights).should eq(expected_outs)
    end
  end
end
