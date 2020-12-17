require "./../../../spec_helper"
require "./../../../spectator_helper"

# add simple RNN functionality
# - [x] Phase 1: initial prototype/skeleton (w/ basic structure calc's)
# - [ ] Phase 2: expand proyotype/skeleton (to include the associated 'mini_net' objects) (in a later PR)
# - [ ] Phase 3: expand proyotype/skeleton (to include the methods to 'guess' and 'train') (in a later PR)

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
  describe "#initialize" do
    context "when NOT passing in any values" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

      it "just some debugging" do
        puts rnn_simple.to_pretty_json

        # rnn_simple.nodal_layer_indexes.map do |li|
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            # debug_info = {"li": li, "ti": ti, "rnn_simple.node_input_sizes[li][ti]": rnn_simple.node_input_sizes[li][ti]}
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(Symbol, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(Symbol, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    3,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 3,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 3,
                "previous_time_column":    rnn_simple.output_size, # 1
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
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(hidden_size_given: hidden_size_given) }

      it "just some debugging" do
        puts rnn_simple.to_pretty_json

        # rnn_simple.nodal_layer_indexes.map do |li|
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(Symbol, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(Symbol, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 10,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.output_size, # 1
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
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "just some debugging" do
        puts rnn_simple.to_pretty_json

        # rnn_simple.nodal_layer_indexes.map do |li|
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(Symbol, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(Symbol, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 10,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.hidden_size, # 1
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 10,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.output_size, # 1
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
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "just some debugging" do
        puts rnn_simple.to_pretty_json

        # rnn_simple.nodal_layer_indexes.map do |li|
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end

      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(Symbol, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(Symbol, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        let(expected_slis) {
          [
            [
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    10,
              },
              {
                "previous_synaptic_layer": rnn_simple.input_size, # 2,
                "previous_time_column":    10,
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 10,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.hidden_size, # 1
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.hidden_size, # 1
              },
            ],
            [
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # 10,
                "previous_time_column":    0,
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.output_size, # 1
              },
              {
                "previous_synaptic_layer": rnn_simple.hidden_size, # ,
                "previous_time_column":    rnn_simple.output_size, # 1
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
  end

end
