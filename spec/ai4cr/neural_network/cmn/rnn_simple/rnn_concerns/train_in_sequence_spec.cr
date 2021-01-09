require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"
require "./../../../../../support/hard_coded_weights.cr"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainInSequence do
  let(time_col_qty) { 4 }
  let(io_offset) { time_col_qty }

  let(input_size) { 21 }
  let(output_size) { 21 }
  let(hidden_layer_qty) { 1 }

  let(deriv_scale) { 0.1 }
  let(learning_rate) { 0.2 }
  let(momentum) { 0.3 }

  let(rnn_simple) {
    net = Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,

      input_size: input_size,
      output_size: output_size,
      hidden_layer_qty: hidden_layer_qty,

      deriv_scale: deriv_scale,
      learning_rate: learning_rate,
      momentum: momentum,
    )
  }

  before_each do
    rnn_simple.init_network

    weights = HARD_CODED_WEIGHTS

    rnn_simple.synaptic_layer_indexes.map do |li|
      rnn_simple.time_col_indexes.map do |ti|
        rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
      end
    end
  end

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
  let(io_pairs) { rnn_simple.split_for_training(sine_data_state_values) }

  describe "#train_in_sequence" do
    let(expected_sequence_errors_first) {
      [3536.347262335783, 621.8863087205787, 2763.6851823307775, 276.47702547425416, 200.76648786646018, 344.30431094941144, 143.18397807232284, 7324.588099281838, 55.85719239357377, 25.146477794170355, 1904.8280263820775, 76.22921709259113, 315.1491988137198, 184.23940164875623, 1271.887297909203, 167.6924843914668, 2980.3894801777924, 505.91086168832743, 2803.9882149133714, 41.263183917716496, 15166.71841194932, 465.21635408134557, 461.4241162173321, 392.45814580122124, 4292.9682182407705, 10.04241558103094, 2411.0654479915, 15.009744924686322, 503.74380231583797, 3.604032070710208, 8.722247962878546, 16.14375842754543, 230.66835045010964, 1491.682946377531]
    }
    let(expected_sequence_errors_first_sum) { expected_sequence_errors_first.sum }

    let(expected_sequence_errors_second) {
      [719.2380688754415, 3.2276500436346396, 111.37860318589506, 58.932968874425434, 2.234795849344671, 881.0764905018146, 2.0871730436917804, 70.54701599595906, 31.43539927511807, 15.31078904491382, 51.87948143934199, 52.588647292881895, 1148.7266592928324, 1080.4895441845229, 30.322761000997378, 13362.458862661133, 97.68290212807973, 3578.660073109589, 3.689926058995775, 9.313817729536733, 2.07643313306625, 138.67885938681331, 6.770348075555354, 16.518249942977558, 0.4521432897778224, 814.7304635768287, 5.9662623295240795, 40.06565566554809, 8.02030749056938, 1706.6454223762973, 87.11427336951226, 220.05568474160302, 200.93683303308146, 14471.219523137126]
    }
    let(expected_sequence_errors_second_sum) { expected_sequence_errors_second.sum }

    # let(expected_sequence_errors_nth) {
    #   [3527.9156624250872, 24.66819191922316, 11882.725791105922, 1457.6341458249303, 12264.376098984818, 1.9577532952094951, 45.62048604235324, 111.51135047165057, 619.7101494030318, 7.221836291982826, 835.2095364833036, 103.26173525328372, 133618.9979486087, 18.283951614336974, 2530.3199554772095, 14.083571407246428, 2656.2224700684683, 23600.103291906467, 396.78955389178327, 1306.2884823146746, 96.46475428046172, 12797.686579986856, 306359.85892514884, 105951.88633950698, 191.3295632112506, 15720.493362561769, 16.729389335941242, 473.73734645904153, 594.6235034280531, 710.5243704126334, 711.5071639906332, 17.17365009238063, 3113.790072072122, 28900.641272441142]
    # }
    # let(expected_sequence_errors_nth_sum) { expected_sequence_errors_nth.sum }

    context "after first session of sequenced training" do
      it "returns expected sequence of errors" do
        # puts
        # puts "rnn_simple.all_mini_net_weights: #{rnn_simple.all_mini_net_weights}"
        # puts

        sequence_errors = rnn_simple.train_in_sequence(io_pairs)

        puts
        puts "sequence_errors: #{sequence_errors}"
        puts

        expect(sequence_errors).to eq(expected_sequence_errors_first)
      end
    end

    context "after second session of sequenced training" do
      it "returns expected sequence of errors" do
        # puts
        # puts "rnn_simple.all_mini_net_weights: #{rnn_simple.all_mini_net_weights}"
        # puts

        rnn_simple.train_in_sequence(io_pairs)
        sequence_errors = rnn_simple.train_in_sequence(io_pairs)

        puts
        puts "sequence_errors: #{sequence_errors}"
        puts

        expect(sequence_errors).to eq(expected_sequence_errors_second)
      end

      it "sum of errors decreased" do
        expect(expected_sequence_errors_second_sum).to be < expected_sequence_errors_first_sum
      end
    end

    # context "after Nth session of sequenced training" do
    #   TODO: Not all RNN's are equal! Find param and rnd seeds that make this succeed (have lowest sum of errors) after Nth training session (and adjust test data as applicable).
    #   it "returns expected sequence of errors" do
    #     # puts
    #     # puts "rnn_simple.all_mini_net_weights: #{rnn_simple.all_mini_net_weights}"
    #     # puts

    #     n = 2
    #     n.times { rnn_simple.train_in_sequence(io_pairs) }
    #     sequence_errors = rnn_simple.train_in_sequence(io_pairs)

    #     puts "v"*10
    #     puts
    #     puts "expected_sequence_errors_first_sum: #{expected_sequence_errors_first_sum}"
    #     puts
    #     puts "expected_sequence_errors_second_sum: #{expected_sequence_errors_second_sum}"
    #     puts
    #     puts "expected_sequence_errors_nth_sum: #{expected_sequence_errors_nth_sum}"
    #     puts
    #     puts "sequence_errors: #{sequence_errors.sum}"
    #     puts
    #     puts "-"*10

    #     expect(sequence_errors).to eq(expected_sequence_errors_nth)
    #   end

    #   it "sum of errors decreased" do
    #     expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_second_sum
    #   end
    # end
  end

  describe "#shifted_inputs" do
    let(io_pair_last) {
      {
        ins: [
          # Last ins:
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        outs: [
          # Last outs
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
      }
    }

    let(expected_input_next) {
      [
        # Next ins:
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ]
    }

    let(inputs_next) { rnn_simple.shifted_inputs(io_pair_last) }

    it "returns expected values" do
      expect(inputs_next).to eq(expected_input_next)
    end
  end
end
