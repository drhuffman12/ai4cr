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

  ins_a = [0.1,0.2,0.3]
  simple_weights_a = [[0.5, -0.5], [-1.0, 1.0], [-0.25, 0.25]]
  expected_outs_a = [-0.225, 0.225]

  # ins_b = [0.2,0.3,0.4,0.5]
  # simple_weights_b = [[-0.25, 0.25], [0.5, -0.5], [-1.0, 1.0], [-0.1, 0.1]]
  # expected_outs_b = [-0.35, 0.35]

  ins_b = [0.2,0.3,0.4]
  simple_weights_b = [[-0.25, 0.25], [0.5, -0.5], [-1.0, 1.0]]
  expected_outs_b = [-0.3, 0.3]

  # ins_b = [0.4]
  # simple_weights_b = [[-1.0, 1.0]]
  # expected_outs_b = [-0.3, 0.3]

  ins_c = [0.5]
  simple_weights_c = [[-0.1, 0.1]]
  expected_outs_c = [-0.05, 0.05]

  describe ".simple_weights_sum" do
    describe "returns expected values" do
      it "example A" do
        # subject.simple_weights_sum(ins_a, simple_weights_a).should eq(expected_outs_a)
        outs = subject.simple_weights_sum(ins_a, simple_weights_a)
        assert_approximate_equality_of_nested_list(expected_outs_a, outs, 0.001)
      end
  
      it "example B" do
        outs = subject.simple_weights_sum(ins_b, simple_weights_b)
        assert_approximate_equality_of_nested_list(expected_outs_b, outs, 0.001)
      end
  
      it "example C" do
        outs = subject.simple_weights_sum(ins_c, simple_weights_c)
        assert_approximate_equality_of_nested_list(expected_outs_c, outs, 0.001)
      end
    end
  end

  describe ".simple_weights_sum_multi" do
    it "returns expected values" do
      ins_list = [ins_a, ins_b, ins_c]
      simple_weights_list = [simple_weights_a, simple_weights_b, simple_weights_c]
      expected_outs_list = [expected_outs_a, expected_outs_b, expected_outs_c]

      outs = subject.simple_weights_sum_multi(ins_list, simple_weights_list)
      assert_approximate_equality_of_nested_list(expected_outs_list, outs, 0.001)
    end
  end

  describe ".simple_weights_sum_multi_sum" do
    it "returns expected values" do
      ins_list = [ins_a, ins_b, ins_c]
      simple_weights_list = [simple_weights_a, simple_weights_b, simple_weights_c]
      expected_outs = [-0.575, 0.575]

      outs = subject.simple_weights_sum_multi_sum(ins_list, simple_weights_list)
      assert_approximate_equality_of_nested_list(expected_outs_c, outs, 0.001)
    end
  end

  describe ".simple_weights_propagate_0_to_1" do
    it "returns propagated simple output values as expected" do
      ins_list = [[1.0]]
      simple_weights_list = [[[-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0]]]
      expected_outs = [0.12,0.27,0.44,0.5,0.56,0.73,0.88]
      diff_radius = 0.01
  
      outs_list = subject.simple_weights_propagate_0_to_1(ins_list, simple_weights_list)
      assert_approximate_equality_of_nested_list(expected_outs, outs_list, diff_radius)
    end
  end

  describe ".simple_weights_propagate_neg_1_to_1" do
    it "returns propagated simple output values as expected" do
      ins_list = [[1.0]]
      simple_weights_list = [[[-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0]]]
      expected_outs = [-0.96,-0.76,-0.25,0.0,0.25,0.76,0.96]
  
      outs_list = subject.simple_weights_propagate_neg_1_to_1(ins_list, simple_weights_list)
      assert_approximate_equality_of_nested_list(expected_outs, outs_list, 0.01)
    end
  end

  describe ".simple_output_errors" do
    it "returns propagated simple output values as expected" do
      expected_outs = [-0.96,-0.76,-0.25,0.0,0.25,0.76,0.96]
      actual_outs = [-0.71, -0.51, 0.0, 0.25, 0.5, 1.01, 1.21]
      expected_errors = [-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25]
  
      error_values = subject.simple_output_errors(expected_outs, actual_outs)
      assert_approximate_equality_of_nested_list(expected_errors, error_values, 0.01)
    end
  end

  describe ".simple_output_deltas_0_to_1" do
    it "returns propagated simple output values as expected" do
      expected_outs = [-0.96,-0.76,-0.25,0.0,0.25,0.76,0.96]
      actual_outs = [-0.71, -0.51, 0.0, 0.25, 0.5, 1.01, 1.21]
      expected_deltas = [0.303,0.19,0.0,-0.05,-0.06,0.0,0.06]
  
      delta_values = subject.simple_output_deltas_0_to_1(expected_outs, actual_outs)
      assert_approximate_equality_of_nested_list(expected_deltas, delta_values, 0.01)
    end
  end

  describe ".simple_output_deltas_neg_1_to_1" do
    it "returns propagated simple output values as expected" do
      expected_outs = [-0.96,-0.76,-0.25,0.0,0.25,0.76,0.96]
      actual_outs = [-0.71, -0.51, 0.0, 0.25, 0.5, 1.01, 1.21]
      expected_deltas = [-0.12,-0.18,-0.25,-0.23,-0.19,0.01,0.12]
  
      delta_values = subject.simple_output_deltas_neg_1_to_1(expected_outs, actual_outs)
      assert_approximate_equality_of_nested_list(expected_deltas, delta_values, 0.01)
    end
  end

  describe ".simple_hidden_delta_from_output_0_to_1" do
    it "returns propagated simple error sum values as expected" do
      ins_a = [0.1,0.2,0.3]
      simple_weights_a = [[0.5, -0.5], [-1.0, 1.0], [-0.25, 0.25]]
      delta_outs_a = [0.1,0.9]

      expected_deltas = [-0.036, 0.128, 0.042]
      
      delta_values = subject.simple_hidden_delta_from_output_0_to_1(ins_a, simple_weights_a, delta_outs_a)
      assert_approximate_equality_of_nested_list(expected_deltas, delta_values, 0.01)
    end
  end

  describe ".simple_hidden_delta_from_output_neg_1_to_1" do
    it "returns propagated simple error sum values as expected" do
      ins_a = [0.1,0.2,0.3]
      simple_weights_a = [[0.5, -0.5], [-1.0, 1.0], [-0.25, 0.25]]
      delta_outs_a = [0.1,0.9]

      expected_deltas = [-0.396, 0.768, 0.182]
      
      delta_values = subject.simple_hidden_delta_from_output_neg_1_to_1(ins_a, simple_weights_a, delta_outs_a)
      assert_approximate_equality_of_nested_list(expected_deltas, delta_values, 0.01)
    end
  end

  describe ".update_simple_weights" do
    learning_rate = 0.5 # : Float64
    momentum = 0.5 # : Float64
    ins = [0.1,0.2,0.3] # : NodesSimple
    simple_weights = [[0.5, -0.5], [-1.0, 1.0], [-0.25, 0.25]] # : WeightsSimple
    simple_weight_changes = [[0.1, -0.1], [-0.2, 0.2], [-0.3, 0.3]]
    delta_outs = [-0.396, 0.768, 0.182]
    updates = subject.update_simple_weights(learning_rate, momentum, ins, simple_weights, simple_weight_changes, delta_outs)

    it "updates simple_weights as expected" do
      expected_simple_weights = [[0.53, -0.51], [-1.14, 1.18], [-0.46, 0.52]]
      
      assert_approximate_equality_of_nested_list(expected_simple_weights, updates[:simple_weights], 0.01)
    end

    it "updates simple_weight_changes as expected" do
      expected_simple_weight_changes = [[-0.04, 0.08], [-0.08, 0.15], [-0.12, 0.23]]
      
      assert_approximate_equality_of_nested_list(expected_simple_weight_changes, updates[:simple_weight_changes], 0.01)
    end
  end

  describe ".propagation_function_0_to_1" do
    map_from_to_approx = {
      -2.0  => 0.12,
      -1.0  => 0.27,
      -0.25 => 0.44,
       0.0  => 0.5,
       0.25 => 0.56,
       1.0  => 0.73,
       2.0  => 0.88,
    }

    map_from_to_approx.each do |value_raw, value_limited|
      it "maps #{value_raw} to approximately #{value_limited}" do
        out_val = subject.propagation_function_0_to_1.call(value_raw)
        assert_approximate_equality(out_val, value_limited, 0.01)
      end
    end
  end

  describe ".propagation_function_neg_1_to_1" do
    map_from_to_approx = {
      -2.0  => -0.96,
      -1.0  => -0.76,
      -0.25 => -0.25,
       0.0  =>  0.0,
       0.25 =>  0.25,
       1.0  =>  0.76,
       2.0  =>  0.96,
    }

    map_from_to_approx.each do |value_raw, value_limited|
      it "maps #{value_raw} to approximately #{value_limited}" do
        out_val = subject.propagation_function_neg_1_to_1.call(value_raw)
        assert_approximate_equality(out_val, value_limited, 0.01)
      end
    end
  end

  describe ".derivative_propagation_function_0_to_1" do
    map_from_to_approx = {
      -2.0  => -6.0,
      -1.0  => -2.0,
      -0.25 => -0.31,
       0.0  =>  0.0,
       0.25 =>  0.19,
       0.5  =>  0.25,
       1.0  =>  0.0,
       2.0  => -2.0,
    }

    map_from_to_approx.each do |value_raw, value_limited|
      it "maps #{value_raw} to approximately #{value_limited}" do
        out_val = subject.derivative_propagation_function_0_to_1.call(value_raw)
        assert_approximate_equality(out_val, value_limited, 0.01)
      end
    end
  end

  describe ".derivative_propagation_function_neg_1_to_1" do
    map_from_to_approx = {
      -2.0  => -3.0,
      -1.0  =>  0.0,
      -0.25 =>  0.94,
       0.0  =>  1.0,
       0.25 =>  0.94,
       1.0  =>  0.0,
       2.0  => -3.0,
    }

    map_from_to_approx.each do |value_raw, value_limited|
      it "maps #{value_raw} to approximately #{value_limited}" do
        out_val = subject.derivative_propagation_function_neg_1_to_1.call(value_raw)
        assert_approximate_equality(out_val, value_limited, 0.01)
      end
    end
  end

end
