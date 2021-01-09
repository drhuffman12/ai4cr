require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainInSequence do
  let(steps) { 20 }
  let(scale) { 100 }
  let(noise_delta_t) { 0.1 }
  let(noise_delta_y) { 0.1 }
  let(noise_offset_t) { 0.5 }

  let(sine_data) {
    (0..2*steps).to_a.map do |i|
      theta = (2 * Math::PI * (i / steps.to_f))
      alt = Math.sin(theta)
      ((alt + 1) / (2))
    end
  }

  let(time_col_qty) { 4 }
  let(io_offset) { time_col_qty }

  let(input_size) { 1 }
  let(output_size) { 1 }
  let(hidden_layer_qty) { 1 }

  let(rnn_simple) {
    Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,

      input_size: input_size,
      output_size: output_size,
      hidden_layer_qty: hidden_layer_qty,
    )
  }

  let(io_pairs) { rnn_simple.split_for_training(sine_data) }

  describe "#shifted_inputs" do
    let(io_pair_last) {
      {
        ins: [
          0.09549150281252639,
          0.02447174185242329,
          0.0,
          0.024471741852423123,
        ],
        outs: [
          0.09549150281252616,
          0.20610737385376326,
          0.34549150281252605,
          0.4999999999999998,
        ],
      }
    }

    let(expected_input_next) {
      [
        0.02447174185242329,
        0.0,
        0.024471741852423123,
        0.09549150281252616,
      ]
    }

    let(inputs_next) { rnn_simple.shifted_inputs(io_pair_last) }

    it "returns expected values" do
      expect(inputs_next).to eq(expected_input_next)
    end
  end
end
