require "./../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::PropsAndInits do
  describe "#initialize" do
    context "when NOT passing in any values" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new }

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    3,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.output_size,
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_simple.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_simple.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_simple.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_simple.hidden_layer_qty).to eq(1)
        end

        it "hidden_size" do
          expect(rnn_simple.hidden_size).to eq(3)
        end

        it "synaptic_layer_qty" do
          expect(rnn_simple.synaptic_layer_qty).to eq(2)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_simple.synaptic_layer_indexes).to eq([0, 1])
        end

        it "@node_input_sizes" do
          expect(rnn_simple.node_input_sizes).to eq(expected_slis)
        end
      end
    end

    context "when passing in hidden_size_given of 10" do
      let(hidden_size_given) { 10 }
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.output_size,
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_simple.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_simple.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_simple.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_simple.hidden_layer_qty).to eq(1)
        end

        it "hidden_size" do
          expect(rnn_simple.hidden_size).to eq(hidden_size_given)
        end

        it "synaptic_layer_qty" do
          expect(rnn_simple.synaptic_layer_qty).to eq(2)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_simple.synaptic_layer_indexes).to eq([0, 1])
        end

        it "@node_input_sizes" do
          expect(rnn_simple.node_input_sizes).to eq(expected_slis)
        end
      end
    end

    context "when passing in hidden_layer_qty of 2, hidden_size_given of 10" do
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.hidden_size,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.output_size,
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_simple.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_simple.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_simple.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_simple.hidden_layer_qty).to eq(hidden_layer_qty)
        end

        it "hidden_size" do
          expect(rnn_simple.hidden_size).to eq(hidden_size_given)
        end

        it "synaptic_layer_qty" do
          expect(rnn_simple.synaptic_layer_qty).to eq(hidden_layer_qty + 1)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_simple.synaptic_layer_indexes).to eq((0..hidden_layer_qty).to_a)
        end

        it "@node_input_sizes" do
          expect(rnn_simple.node_input_sizes).to eq(expected_slis)
        end
      end
    end

    context "when passing in time_col_qty of 3, hidden_layer_qty of 2, hidden_size_given of 10" do
      let(time_col_qty) { 3 }
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    10,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.hidden_size,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.hidden_size,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.output_size,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size,
                "previous_time_column":    rnn_simple.output_size,
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_simple.time_col_qty).to eq(time_col_qty)
        end

        it "input_size" do
          expect(rnn_simple.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_simple.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_simple.hidden_layer_qty).to eq(hidden_layer_qty)
        end

        it "hidden_size" do
          expect(rnn_simple.hidden_size).to eq(hidden_size_given)
        end

        it "synaptic_layer_qty" do
          expect(rnn_simple.synaptic_layer_qty).to eq(hidden_layer_qty + 1)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_simple.synaptic_layer_indexes).to eq((0..hidden_layer_qty).to_a)
        end

        it "@node_input_sizes" do
          expect(rnn_simple.node_input_sizes).to eq(expected_slis)
        end
      end
    end

    # it "just some debugging" do # TODO: REMOVE before merging!
    #   # puts rnn_simple.to_pretty_json

    #   # rnn_simple.nodal_layer_indexes.map do |li|
    #   rnn_simple.synaptic_layer_indexes.map do |li|
    #     rnn_simple.time_col_indexes.map do |ti|
    #       # debug_info = {"li": li, "ti": ti, "rnn_simple.node_input_sizes[li][ti]": rnn_simple.node_input_sizes[li][ti]}
    #       debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
    #       # puts debug_info.to_json
    #     end
    #   end
    # end

    context "mini_net_set" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new }

      it "each are of the expected width and height" do
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            mini_net = rnn_simple.mini_net_set[li][ti]

            expected_input_size = rnn_simple.node_input_sizes[li][ti].values.sum
            expect(mini_net.height).to eq(expected_input_size)

            expected_output_size = rnn_simple.node_output_sizes[li]
            expect(mini_net.width).to eq(expected_output_size)
          end
        end
      end
    end
  end
end
