require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::DataUtils do
  let(time_col_qty) { 4 }
  let(io_offset) { time_col_qty }

  let(input_size) { 21 }
  let(output_size) { 21 }
  let(hidden_layer_qty) { 1 }

  let(rnn_simple) {
    Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,

      input_size: input_size,
      output_size: output_size,
      hidden_layer_qty: hidden_layer_qty,
    )
  }

  let(steps) { 20 }
  let(scale) { 100 }

  let(sine_data) {
    (0..2*steps).to_a.map do |i|
      theta = (2 * Math::PI * (i / steps.to_f))
      alt = Math.sin(theta)
      # ((alt + 1) / (2))
      alt
    end
  }

  let(sine_data_state_values) { rnn_simple.float_to_state_values(sine_data) }

  # let(expected_sine_data_state_values) {
  #   [
  #     # First ins:
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],

  #     # First outs:
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],

  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

  #     # Last ins:
  #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

  #     # Last outs
  #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #   ]
  # }

  describe "#indexes_for_training_and_eval" do
    context "given sine_data_state_values" do
      let(training_and_eval_indexes) { rnn_simple.indexes_for_training_and_eval(sine_data_state_values) }

      context "returns" do
        let(expected_training_data_size_size) { sine_data_state_values.size - io_offset - time_col_qty + 1 }

        it "expected type" do
          expect(training_and_eval_indexes).to be_an(Ai4cr::NeuralNetwork::Rnn::RnnTrainingIndexes)
        end

        context "expected metadata for key" do
          it ":training_in_indexes" do
            expect(training_and_eval_indexes[:training_in_indexes].size).to eq(expected_training_data_size_size)
          end

          it ":training_out_indexes" do
            expect(training_and_eval_indexes[:training_out_indexes].size).to eq(expected_training_data_size_size)
          end
        end

        context "expected values for key" do
          let(expected_training_in_indexes) {
            [
              {i_from: 0, i_to: 3}, {i_from: 1, i_to: 4}, {i_from: 2, i_to: 5}, {i_from: 3, i_to: 6},
              {i_from: 4, i_to: 7}, {i_from: 5, i_to: 8}, {i_from: 6, i_to: 9}, {i_from: 7, i_to: 10},
              {i_from: 8, i_to: 11}, {i_from: 9, i_to: 12}, {i_from: 10, i_to: 13}, {i_from: 11, i_to: 14},
              {i_from: 12, i_to: 15}, {i_from: 13, i_to: 16}, {i_from: 14, i_to: 17}, {i_from: 15, i_to: 18},
              {i_from: 16, i_to: 19}, {i_from: 17, i_to: 20}, {i_from: 18, i_to: 21}, {i_from: 19, i_to: 22},
              {i_from: 20, i_to: 23}, {i_from: 21, i_to: 24}, {i_from: 22, i_to: 25}, {i_from: 23, i_to: 26},
              {i_from: 24, i_to: 27}, {i_from: 25, i_to: 28}, {i_from: 26, i_to: 29}, {i_from: 27, i_to: 30},
              {i_from: 28, i_to: 31}, {i_from: 29, i_to: 32}, {i_from: 30, i_to: 33}, {i_from: 31, i_to: 34},
              {i_from: 32, i_to: 35}, {i_from: 33, i_to: 36},
            ]
          }
          let(expected_training_out_indexes) {
            [
              {i_from: 4, i_to: 7}, {i_from: 5, i_to: 8}, {i_from: 6, i_to: 9}, {i_from: 7, i_to: 10},
              {i_from: 8, i_to: 11}, {i_from: 9, i_to: 12}, {i_from: 10, i_to: 13}, {i_from: 11, i_to: 14},
              {i_from: 12, i_to: 15}, {i_from: 13, i_to: 16}, {i_from: 14, i_to: 17}, {i_from: 15, i_to: 18},
              {i_from: 16, i_to: 19}, {i_from: 17, i_to: 20}, {i_from: 18, i_to: 21}, {i_from: 19, i_to: 22},
              {i_from: 20, i_to: 23}, {i_from: 21, i_to: 24}, {i_from: 22, i_to: 25}, {i_from: 23, i_to: 26},
              {i_from: 24, i_to: 27}, {i_from: 25, i_to: 28}, {i_from: 26, i_to: 29}, {i_from: 27, i_to: 30},
              {i_from: 28, i_to: 31}, {i_from: 29, i_to: 32}, {i_from: 30, i_to: 33}, {i_from: 31, i_to: 34},
              {i_from: 32, i_to: 35}, {i_from: 33, i_to: 36}, {i_from: 34, i_to: 37}, {i_from: 35, i_to: 38},
              {i_from: 36, i_to: 39}, {i_from: 37, i_to: 40},
            ]
          }
          let(expected_eval_ins) {
            {i_from: 34, i_to: 37}
          }

          it ":training_in_indexes" do
            expect(training_and_eval_indexes[:training_in_indexes]).to eq(expected_training_in_indexes)
          end

          it ":training_out_indexes" do
            expect(training_and_eval_indexes[:training_out_indexes]).to eq(expected_training_out_indexes)
          end

          it ":next_eval_in_indexes" do
            expect(training_and_eval_indexes[:next_eval_in_indexes]).to eq(expected_eval_ins)
          end
        end
      end
    end
  end

  describe "#float_to_state_values" do
    let(to_min_i) { 0 }
    let(to_max_i) { 10 }
    let(from_min) { 0.0 }

    let(input_size) { to_max_i - to_min_i + 1 }
    let(output_size) { to_max_i - to_min_i + 1 }
    let(hidden_layer_qty) { 1 }

    let(value_states) {
      rnn_simple.float_to_state_values(
        values: sine_data,
        to_min_i: to_min_i,
        to_max_i: to_max_i,
        from_min: from_min
      )
    }

    context "returns" do
      it "same outer array size" do
        expect(value_states.size).to eq(sine_data.size)
      end

      context "inner array type of" do
        context "sine_data is" do
          it "single float value" do
            expect(sine_data.first).to be_a(Float64)
          end
        end

        context "value_states is" do
          it "array of float values" do
            expect(value_states.first).to be_a(Array(Float64))
          end
        end
      end
    end
  end
end
